import math
from scipy.io import loadmat
import numpy as np


def snake_grid(linear_vector, shape, facing_down=True):
    snaked = linear_vector.reshape((shape[1], shape[0]))
    snaked[1::2] = snaked[1::2, ::-1]
    snaked = snaked.T
    if facing_down:
        return snaked
    else:
        return np.fliplr(np.flipud(snaked))


def snake_data(linear_vector, shape, facing_down=True):
    linear_vector = linear_vector.T
    snaked = linear_vector.reshape((shape[1], shape[0], -1))
    snaked[1::2] = snaked[1::2, ::-1, :]
    snaked = np.moveaxis(snaked, 0, 1)
    if facing_down:
        return snaked
    else:
        return snaked[::-1, :, :]


def unsnake_data(snaked: np.ndarray, facing_down=True):
    if not facing_down:
        snaked = snaked[::-1, :, :]
    unsnaked = np.moveaxis(snaked, 0, 1)
    unsnaked[1::2] = unsnaked[1::2, ::-1, :]
    return unsnaked.reshape(-1, (unsnaked.shape[-1])).T


def averaged_grid_data(snaked, window_shape=(2, 2)):
    windowed = np.lib.stride_tricks.sliding_window_view(
        snaked, window_shape=window_shape, axis=(0, 1)
    )
    return np.mean(windowed, axis=(-2, -1))


def create_centered_mask(patch_mask, shape):
    mask_height, mask_width = patch_mask.shape
    shape_height, shape_width = shape
    center_y, center_x = mask_height // 2, mask_width // 2
    y_start = center_y - shape_height // 2
    y_end = y_start + shape_height
    x_start = center_x - shape_width // 2
    x_end = x_start + shape_width
    centered_mask = np.zeros_like(patch_mask)
    centered_mask[y_start:y_end, x_start:x_end] = 1

    return centered_mask


class HDEMG2Bipolar:
    def __init__(
        self,
        data,
        patch_shape=(13, 5),
        radius=5.7,
        fov=95.0,
        channel_mask=None,
        facing_down=True,
        average_filter=(None, None),
        min_r=0,
        reduced_patch_shape=None,
    ):
        """
        Automatically generates single-differential samples in the grid based on criteria on distance and orientation.
        Can be iterated through, call list(HDEMG2Bipolar) to calculate all combinations. Set

        :param data: HDEMG data in shape (time, channel). Assuming snaking pattern filling the grid.
        :param patch_shape: Grid dimensions, (rows, columns)
        :param radius: Maximum distance between two electrodes for differential sampling
        :param fov: 'Field of view', sampling axis can deviate from longitudinal axis in fov/2 degrees either way
        :param channel_mask: Optional vector in the shape (1, channel), allows for ignoring channels (e.g. if they are
        dead, or other conditions)
        """
        print("Facing down is", facing_down)
        self.channel_offset = np.prod(patch_shape) - data.shape[-1]
        self.original_patch_shape = patch_shape
        self.original_n_channels = min(
            data.shape[-1], np.prod(self.original_patch_shape)
        )
        self.min_r = min_r

        if any([e is not None for e in average_filter]):
            zero_padded_data = np.concatenate(
                (np.zeros((data.shape[0], self.channel_offset)), data), axis=1
            )
            self.HDEMGData = unsnake_data(
                averaged_grid_data(
                    snake_data(zero_padded_data, patch_shape, facing_down),
                    average_filter,
                )
            )[:, self.channel_offset :]

            self.patch_shape = (
                patch_shape[0] - average_filter[0] + 1,
                patch_shape[1] - average_filter[1] + 1,
            )
        else:
            self.HDEMGData = data
            self.patch_shape = patch_shape  # this will determine whether we are using the 13x5 or the 8x8

        self.n_channels = min(self.HDEMGData.shape[-1], np.prod(self.patch_shape))

        # Calculate how many "blank" placeholder channels there are as the mismatch from

        self.radius = radius  # max distance
        self.fov = fov
        if channel_mask is None:
            channel_mask = np.ones(self.n_channels, dtype=int)

        self.region_of_interest = self.get_region_of_interest()

        self.channel_mask = np.concatenate(
            (np.zeros(self.channel_offset, dtype=int), channel_mask)
        )
        self.patch_mask = snake_grid(
            self.channel_mask, self.patch_shape, facing_down=facing_down
        )
        if reduced_patch_shape:
            self.patch_mask = create_centered_mask(self.patch_mask, reduced_patch_shape)

        # ndarray in the shape of the patch, with the electrode numbering as given on the patch
        self.patch_idx = (
            snake_grid(
                np.concatenate(
                    (
                        np.zeros(self.channel_offset, dtype=int),
                        np.arange(self.n_channels, dtype=int) + 1,
                    )
                ),
                self.patch_shape,
                facing_down=facing_down,
            )
            * self.patch_mask
        )

        self.original_offset_vector = np.ones(
            np.prod(self.original_patch_shape)
        ).astype(int)
        self.original_offset_vector[0] = int(0)
        self.original_patch_idx = snake_grid(
            np.concatenate(
                (
                    np.zeros(self.channel_offset, dtype=int),
                    np.arange(self.original_n_channels, dtype=int) + 1,
                )
            ),
            self.original_patch_shape,
            facing_down=facing_down,
        )

        def sampling_sort(channel_tuple):
            return channel_tuple[0] * 1000 + channel_tuple[1]

        # NOTE: Channel numbering starts at 1
        self.list_of_combinations = sorted(
            list(self.get_bipolar_combinations()), key=sampling_sort
        )
        self.bipolar_data = self.get_bipolar_data()
        self.current = 0

    @property
    def n_rows(self):
        return self.patch_shape[0]

    @property
    def n_cols(self):
        return self.patch_shape[1]

    def get_bipolar_combinations(self):
        combination_set = set()

        for j, i in np.argwhere(self.patch_mask.T):
            sampled_indices = self.sample_range(i, j)
            combination_set.update(
                {
                    tuple(sorted((self.patch_idx[i, j], to_idx)))
                    for to_idx in self.patch_idx[
                        sampled_indices[:, 0], sampled_indices[:, 1]
                    ]
                    if self.channel_mask[to_idx - 1 + self.channel_offset]
                }
            )
        return combination_set

    def get_bipolar_data(self):
        differential_columns = []
        for i, j in self.list_of_combinations:
            differential = self.HDEMGData[:, i] - self.HDEMGData[:, j]
            differential_columns.append(differential)
        differential_array = np.column_stack(differential_columns)
        return differential_array

    def get_region_of_interest(self):
        fov_rad = self.fov / 180 * np.pi
        upper_half = [
            (r, c)
            for r in range(math.ceil(self.radius))
            for c in range(math.floor(-self.radius), math.ceil(self.radius))
            if self.min_r < (r * r + c * c) < self.radius * self.radius
            and -1 / 2 * fov_rad < (np.arctan2(r, c) - np.pi / 2) < 1 / 2 * fov_rad
        ]
        return np.concatenate(
            (upper_half, -1 * np.array(upper_half))
        )  # Could adapt so upper quadrant was enough

    def sample_range(self, i, j):
        return np.array(
            [
                (r + i, c + j)
                for r, c in self.region_of_interest
                if self.n_rows > r + i >= 0 and self.n_cols > c + j >= 0
            ]
        )

    def show_roi(self):
        """
        Visualisation function to check RoI shape is according to given radius and FoV.
        """
        import matplotlib.pyplot as plt

        roi = self.region_of_interest
        img = np.zeros((roi[:, 0].max() * 2 + 1, roi[:, 1].max() * 2 + 1))

        sampled_indices = np.array(
            [(r + roi[:, 0].max(), c + roi[:, 1].max()) for r, c in roi]
        )
        img[sampled_indices[:, 0], sampled_indices[:, 1]] = 1
        img[roi[:, 0].max(), roi[:, 1].max()] = 0.5
        plt.imshow(
            img,
            extent=[roi[:, 1].min(), roi[:, 1].max(), roi[:, 0].min(), roi[:, 0].max()],
        )
        plt.title("Region of Interest per electrode")
        plt.show()

    def plot_grid(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        for row_idx, row in enumerate(self.patch_idx):
            for col_idx, elem in enumerate(row):
                circle = plt.Circle((col_idx, row_idx), 0.3, ec="k", fc="w")
                ax.add_patch(circle)
                ax.annotate(str(elem), xy=(col_idx, row_idx), ha="center", va="center")

        ax.set_xlim((-0.5, self.patch_shape[1] + 0.5))
        ax.set_ylim((-0.5, self.patch_shape[0] + 0.5))
        ax.axis("equal")

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self.list_of_combinations):
            single_differential = self[self.current]
            self.current += 1
            return single_differential
        raise StopIteration

    def __len__(self):
        """
        Denotes the number of allowed combinations
        """
        return len(self.list_of_combinations)

    def __getitem__(self, combination_index):
        channel1, channel2 = np.atleast_2d(
            self.list_of_combinations[combination_index]
        ).T
        # Convert to 0-indexed
        return (
            self.HDEMGData[:, channel1 - self.channel_offset]
            - self.HDEMGData[:, channel2 - self.channel_offset]
        )


if __name__ == "__main__":
    sampler = HDEMG2Bipolar(
        data=np.zeros((100, 64)),
        radius=3.2,
        fov=95,
        patch_shape=(13, 5),
        reduced_patch_shape=(5, 3),
    )
    print(sampler.patch_idx)
    print(sampler.patch_mask)
    print(sampler.list_of_combinations)
