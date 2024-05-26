import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr
import os



sys.path.append(".")

class Subtile:
    def __init__(
        self,
        image,
        mask,
        slice_size: tuple = (4, 4),
    ):
        self.image = image
        self.mask = mask
        self.slice_size = slice_size
        self.id = str(image.attrs["id"])


    def __calculate_slice_index(self, x: int, y: int, slice_size: tuple, length: tuple):
        # calculate the start and end indices for the slice based
        # on the slice_size and the x and y coordinates
        start_index = (
            int(np.divide(length[0], slice_size[0]) * x),
            int(np.divide(length[1], slice_size[1]) * y),
        )
        end_index = (
            int(np.divide(length[0], slice_size[0]) * (x + 1)),
            int(np.divide(length[1], slice_size[1]) * (y + 1)),
        )

        if start_index[0] > length[0] or start_index[1] > length[1]:
            raise IndexError(
                f"Start index {start_index} out of range for img of shape {length}"
            )

        if end_index[0] > length[0] or end_index[1] > length[1]:
            raise IndexError(
                f"End index {end_index} out of range for img of shape {length}"
            )

        return start_index, end_index

    def get_subtile_from_parent_image(self, x: int, y: int) -> xr.Dataset:
        img_length = (self.image.shape[0], self.image.shape[1])
        label_length = (self.mask.shape[0], self.mask.shape[1])

        start_index_img, end_index_img = self.__calculate_slice_index(
            x, y, self.slice_size, img_length
        )
        start_index_label, end_index_label = self.__calculate_slice_index(
            x, y, self.slice_size, label_length
        )

        sliced_data_array = self.image[
                start_index_img[0] : end_index_img[0],
                start_index_img[1] : end_index_img[1],
                :
            ]
        sliced_data_array.attrs["x"] = x
        sliced_data_array.attrs["y"] = y

        new_ground_truth = self.mask[
            start_index_label[0] : end_index_label[0],
            start_index_label[1] : end_index_label[1],
            :
        ]
        new_ground_truth.attrs["x"] = x
        new_ground_truth.attrs["y"] = y

        return sliced_data_array, new_ground_truth

    def _save_image(
        self, subtiled_data_array: xr.DataArray, subtile_directory: Path, x: int, y: int
    ):
        subtiled_data_array.to_netcdf(
            subtile_directory
            / str(subtiled_data_array.attrs["id"])
            / f"{x}_{y}"
            / f"{x}_{y}.nc"
        )

    def _save_label(
        self,
        subtiled_ground_truth: xr.DataArray,
        subtile_directory: Path,
        x: int,
        y: int,
    ):
        subtiled_ground_truth.to_netcdf(
            subtile_directory
            / str(subtiled_ground_truth.attrs["id"])
            / f"{x}_{y}"
            / f"mask.nc"
        )

    def save(self, directory_to_save: Path) -> None:
        directory_to_save.mkdir(parents=True, exist_ok=True)
        subtile_directory = directory_to_save / "subtiles"
        subtile_directory.mkdir(parents=True, exist_ok=True)
        for x in range(self.slice_size[0]):
            for y in range(self.slice_size[1]):
                subtile, subtiled_mask = (
                    self.get_subtile_from_parent_image(x, y)
                )

                Path(subtile_directory / self.id).mkdir(exist_ok=True)
                assert Path(subtile_directory / self.id).exists()

                Path(subtile_directory / self.id / f"{x}_{y}").mkdir(
                    exist_ok=True
                )
                assert Path(
                    subtile_directory / self.id / f"{x}_{y}"
                ).exists()
                self._save_image(subtile, subtile_directory, x, y)
                self._save_label(subtiled_mask, subtile_directory, x, y)
        self.image = None
        self.mask = None

    @staticmethod
    def load_subtile(
        directory_to_load: Path,
        x: int,
        y: int,
    ):
        tile_dir = directory_to_load
        subtile_file = tile_dir / f"{x}_{y}" / f"{x}_{y}.nc"
        assert subtile_file.exists() == True, f"{subtile_file} does not exist"
        data_array = xr.load_dataarray(subtile_file)
        gt_data_array = xr.load_dataarray(
                tile_dir / f"{x}_{y}" / "mask.nc"
            )
        assert data_array.attrs["x"] == np.int32(x), f"{data_array.attrs['x']}, {x}"
        assert data_array.attrs["y"] == np.int32(y), f"{data_array.attrs['y']}, {y}"
        assert gt_data_array.attrs["x"] == np.int32(x), f"{gt_data_array.attrs['x']}, {x}"
        assert gt_data_array.attrs["y"] == np.int32(y), f"{gt_data_array.attrs['y']}, {y}"
        return data_array, gt_data_array
    


    @staticmethod
    def load_subtile_by_dir(
        directory_to_load: Path,
        slice_size,
        has_gt: bool = True,
    ):  
        path_str = str(directory_to_load).split(os.sep)[-1].split("_")
        x = path_str[0]
        y = path_str[1]
        subtile_file = directory_to_load / f"{x}_{y}.nc"
        assert subtile_file.exists() == True, f"{subtile_file} does not exist"
        data_array = xr.load_dataarray(subtile_file)

        if has_gt:
            gt_data_array = xr.load_dataarray(
                directory_to_load / f"mask.nc"
            )
        else:
            gt_data_array = None

        subtile = Subtile(
            data_array,
            gt_data_array,
            slice_size=slice_size,
        )
        return subtile


    @staticmethod
    def restich(dir, slice_size = (4,4)):
        row_x = []
        row_y = []
        for x in range(slice_size[0]):
            col_x = []
            col_y = []
            for y in range(slice_size[1]):
                data_array_x, data_array_y = Subtile.load_subtile(dir, x, y)
                    # remove the subtile attributes now that they are no longer needed
                del data_array_x.attrs["x"]
                del data_array_x.attrs["y"]
                del data_array_y.attrs["x"]
                del data_array_y.attrs["y"]
                col_x.append(data_array_x)
                col_y.append(data_array_y)
            row_x.append(xr.concat(col_x, dim="width"))
            row_y.append(xr.concat(col_y, dim="width"))
        original_image = xr.concat(row_x, dim="height")
        original_mask = xr.concat(row_y, dim="height")
        return original_image, original_mask