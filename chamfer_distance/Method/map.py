import numpy as np

def createDataMapDict(data_list: list) -> dict:
    if len(data_list) == 0:
        return {}

    if len(data_list) < 2:
        data_array = np.asarray(data_list[0])
    else:
        data_array = np.hstack(data_list)

    sorted_data_array = np.sort(data_array)

    data_map = {}

    for i in range(sorted_data_array.shape[0]):
        data_map[sorted_data_array[i]] = i

    return data_map

def mapData(data: np.ndarray, data_map: dict) -> np.ndarray:
    mapped_data = []

    for d in data:
        mapped_d = data_map[d]

        mapped_data.append(mapped_d)

    mapped_data = np.asarray(mapped_data)

    return mapped_data
