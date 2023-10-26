import numpy as np

def hjorth_parameters(data):
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)

    activity = np.var(data)
    mobility = np.sqrt(np.var(diff1) / activity)
    complexity = np.sqrt(np.var(diff2) / np.var(diff1))

    return np.array([activity, mobility, complexity])


def readfile(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            values = [float(val) for val in line.strip().split()]
            data.append(values)
    return np.array(data, dtype=float)


def savedata(hjorth_params, output_path):
    np.savetxt(output_path, hjorth_params, delimiter='\t', comments='', fmt='%.8f')
    print("Hjorth parameters saved to", output_path)


def calculate_hjorth(filename):
    data = readfile(filename)
    hjorth_params = []

    for row in data:
        hjorth_vector = hjorth_parameters(row)
        hjorth_params.append(hjorth_vector)

    return np.array(hjorth_params)


def feature_extraction_hjorth():
    print("----------FeatureExtraction-Hjorth----------")

    input_filename = "../data/processed/features_raw.dat"
    output_path = "../data/processed/hjorth_parameters.dat"

    hjorth_params = calculate_hjorth(input_filename)

    savedata(hjorth_params, output_path)

    print("--------------------" + "\n")


if __name__ == "__main__":
    feature_extraction_hjorth()
