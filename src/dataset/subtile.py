import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr



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
            / subtiled_ground_truth.attrs["id"]
            / f"{x}_{y}"
            / f"mask.nc"
        )

    def save(self, directory_to_save: Path) -> None:
        directory_to_save.mkdir(parents=True, exist_ok=True)
        subtile_directory = directory_to_save / "subtiles"
        subtile_directory.mkdir(parents=True, exist_ok=True)

        # iterate over the slice_size
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

                # save the subtile of the image at the images directory

                self._save_image(subtile, subtile_directory, x, y)
                # save the subtile of the label at the labels directory
                self._save_label(subtiled_mask, subtile_directory, x, y)
        # clear the data because it has been saved into the subtiled files
        self.image = None
        self.mask = None


    def load_subtile(
        self,
        directory_to_load: Path,
        x: int,
        y: int,
    ) -> List[xr.DataArray]:
        """
        Loads a subtile file ({id}_{x}_{y}.npy)

        Parameters:
            subtile_file: path to the subtile file
        Returns:
            List[xr.DataArray]
        """
        tile_dir = directory_to_load / "subtiles" / str(self.id)
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
        """
        Loads a directory of subtile files ({parent_tile_id}_{x}_{y}.npy)

        Parameters:
            directory_to_load: Path to the subtile file directory
            satellite_type_list: list of satellites to load
            slice_size: slice size of the subtile
        Returns:
            List[xr.DataArray]
        """            
        path_str = str(directory_to_load).split("/")[-1].split("_")
        x = path_str[0]
        y = path_str[1]
        print(str(directory_to_load / f"{x}_{y}.nc"))
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


    def restitch(
        self, directory_to_load: Path
    ) -> None:
        """
        Restitiches the subtile images to their original image

        Parameters:
            directory_to_load: path to the directory where the subtile images and labels are loaded from
            satellite_type_list: list of satellite types that dictates which satellites will be loaded

        Returns:
            result: Tuple containing:
                restitched_image: List[xr.DataArray]
                restitched_label: xr.DataArray
        """
        # even though this is slightly less efficient that iterating over once,
        # it's way more readable and understandable for everyone

        # add the ground truth to the satellite_type_list so this
        # operation below will retrieve it for us
        
        #satellite_type_list_with_gt = satellite_type_list + [SatelliteType.GT]

        list_of_data_array = list()
        for satellite_type in satellite_type_list_with_gt:
            row = []
            for x in range(self.slice_size[0]):
                col = []
                for y in range(self.slice_size[1]):
                    data_array = self.load_subtile(
                        directory_to_load, [satellite_type], x, y
                    )[0]
                    # remove the subtile attributes now that they are no longer needed
                    del data_array.attrs["x"]
                    del data_array.attrs["y"]

                    col.append(data_array)
                row.append(xr.concat(col, dim="width"))
            data_array = xr.concat(row, dim="height")
            list_of_data_array.append(data_array)

        self.satellite_list = list_of_data_array[:-1]
        self.ground_truth = list_of_data_array[-1]