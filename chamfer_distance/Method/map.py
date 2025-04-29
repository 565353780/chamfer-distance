import numpy as np

def createDataMapDict(data: np.ndarray) -> dict:
    if data.shape[0] == 0:
        return {}

    unique_data = np.unique(data)
    sorted_data = np.sort(unique_data)

    data_map = {}

    for i in range(sorted_data.shape[0]):
        data_map[sorted_data[i]] = i

    return data_map

def mapData(data: np.ndarray, data_map: dict) -> np.ndarray:
    mapped_data = []

    for d in data:
        mapped_d = data_map[d]

        mapped_data.append(mapped_d)

    mapped_data = np.asarray(mapped_data)

    return mapped_data
