import pandas as pd
import random
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# oversampling methods





def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=42):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

# trainset_X, trainset_Y = trainset.iloc[:, 1:].to_numpy(), trainset.iloc[:, 0].to_numpy()
# oversample = SMOTE()
# X, y = oversample.fit_resample(trainset_X, trainset_Y)


# input is in dataframe
def SMOTE_oversample(data):
    trainset_X, trainset_Y = data.iloc[:, 1:], data.iloc[:, 0:1]
    oversample = SMOTE()
    X, y = oversample.fit_resample(trainset_X, trainset_Y)
    trainset = pd.concat([y, X], axis=1)
    cancer = trainset.loc[trainset['Cancer'] == 1]
    nocancer = trainset.loc[trainset['Cancer'] == 0]
    return trainset


# input is in dataframe
def BorderlineSMOTE_oversample(data):
    trainset_X, trainset_Y = data.iloc[:, 1:], data.iloc[:, 0:1]
    oversample = BorderlineSMOTE()
    X, y = oversample.fit_resample(trainset_X, trainset_Y)
    trainset = pd.concat([y, X], axis=1)
    cancer = trainset.loc[trainset['Cancer'] == 1]
    nocancer = trainset.loc[trainset['Cancer'] == 0]
    return trainset

# input is in dataframe
def SVMSMOTE_oversample(data):
    trainset_X, trainset_Y = data.iloc[:, 1:], data.iloc[:, 0:1]
    oversample = SVMSMOTE()
    X, y = oversample.fit_resample(trainset_X, trainset_Y)
    trainset = pd.concat([y, X], axis=1)
    cancer = trainset.loc[trainset['Cancer'] == 1]
    nocancer = trainset.loc[trainset['Cancer'] == 0]
    return trainset

# input is in dataframe
def SMOTE_with_random_undersampling_oversample(data):
    trainset_X, trainset_Y = data.iloc[:, 1:], data.iloc[:, 0:1]
    # define pipeline
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(trainset_X, trainset_Y)
    trainset = pd.concat([y, X], axis=1)
    cancer = trainset.loc[trainset['Cancer'] == 1]
    nocancer = trainset.loc[trainset['Cancer'] == 0]
    return trainset


# input is in dataframe
def ADASYN_oversample(data):
    trainset_X, trainset_Y = data.iloc[:, 1:], data.iloc[:, 0:1]
    oversample = ADASYN()
    X, y = oversample.fit_resample(trainset_X, trainset_Y)
    trainset = pd.concat([y, X], axis=1)
    cancer = trainset.loc[trainset['Cancer'] == 1]
    nocancer = trainset.loc[trainset['Cancer'] == 0]
    return trainset

def rotation(data):
    n = data.shape[0]
    for i in tqdm(range(n)):
        row = data.iloc[i]
        L0, L1, L2, L3, L4 = row["L0"], row["L1"], row["L2"], row["L3"], row["L4"]
        L5, L6, L7, L8, L9 = row["L5"], row["L6"], row["L7"], row["L8"], row["L9"]

        LS0, LS1, LS2, LS3, LS4 = row["Skin L0"], row["Skin L1"], row["Skin L2"], row["Skin L3"], row["Skin L4"]
        LS5, LS6, LS7, LS8, LS9 = row["Skin L5"], row["Skin L6"], row["Skin L7"], row["Skin L8"], row["Skin L9"]

        R0, R1, R2, R3, R4 = row["R0"], row["R1"], row["R2"], row["R3"], row["R4"]
        R5, R6, R7, R8, R9 = row["R5"], row["R6"], row["R7"], row["R8"], row["R9"]

        RS0, RS1, RS2, RS3, RS4 = row["Skin R0"], row["Skin R1"], row["Skin R2"], row["Skin R3"], row["Skin R4"]
        RS5, RS6, RS7, RS8, RS9 = row["Skin R5"], row["Skin R6"], row["Skin R7"], row["Skin R8"], row["Skin R9"]

        LT0, LTS0 = row["T0"], row["Skin T0"]
        RT1, RTS1 = row["T1"], row["Skin T1"]

        # 90%
        # new_row = np.array(Cancer	R0	R1	R2	R3	R4	R5	R6	R7	R8	R9	L0	L1	L2	L3	L4	L5	L6	L7	L8	L9	Skin R0	Skin R1	Skin R2	Skin R3	Skin R4	Skin R5	Skin R6	Skin R7	Skin R8	Skin R9	Skin L0	Skin L1	Skin L2	Skin L3	Skin L4	Skin L5	Skin L6	Skin L7	Skin L8	Skin L9	T0	T1	Skin T0	Skin T1)
        new_row_90 = np.array([row["Cancer"], R0, R7, R8, R1, R2, R3, R4, R5, R6, R9,
                            L0, L3, L4, L5, L6, L7, L8, L1, L2, L9,
                            RS0, RS7, RS8, RS1, RS2, RS3, RS4, RS5, RS6, RS9,
                            LS0, LS3, LS4, LS5, LS6, LS7, LS8, LS1, LS2, LS9,
                            LT0, RT1, LTS0, RTS1])
        new_row_180 = np.array([row["Cancer"], R0, R5, R6, R7, R8, R1, R2, R3, R4, R9,
                               L0, L5, L6, L7, L8, L1, L2, L3, L4, L9,
                               RS0, RS5, RS6, RS7, RS8, RS1, RS2, RS3, RS4, RS9,
                               LS0, LS5, LS6, LS7, LS8, LS1, LS2, LS3, LS4, LS9,
                               LT0, RT1, LTS0, RTS1])

        new_row_270 = np.array([row["Cancer"], R0, R3, R4, R5, R6, R7, R8, R1, R2, R9,
                                L0, L7, L8, L1, L2, L3, L4, L5, L6, L9,
                                RS0, RS3, RS4, RS5, RS6, RS7, RS8, RS1, RS2, RS9,
                                LS0, LS7, LS8, LS1, LS2, LS3, LS4, LS5, LS6, LS9,
                                LT0, RT1, LTS0, RTS1])
        # data.append(new_row_90, ignore_index=True)
        # data.append(new_row_180, ignore_index=True)
        # data.append(new_row_270, ignore_index=True)
        data.loc[len(data)] = new_row_90
        data.loc[len(data)] = new_row_180
        data.loc[len(data)] = new_row_270
    return data

# trainset.to_csv("./dataset/train_origin.csv")
# testset.to_csv("./dataset/test_origin.csv")

# trainset = SMOTE_oversample(trainset)
# trainset.to_csv("./dataset/train_SMOTE.csv")

# trainset = SMOTE_with_random_undersampling_oversample(trainset)
# trainset.to_csv("./dataset/train_SMOTE_random_undersample.csv")

# trainset = BorderlineSMOTE_oversample(trainset)
# trainset.to_csv("./dataset/train_BorderlineSMOTE.csv")

# trainset = SVMSMOTE_oversample(trainset)
# trainset.to_csv("./dataset/train_SVMSMOTE.csv")



path = "./dataset/dataset_clean.csv"
df = pd.read_csv(path)

trainset, validset, testset = train_validate_test_split(df, train_percent=0.8, validate_percent=0.1)
trainset = ADASYN_oversample(trainset)
print()

# trainset = rotation(trainset.iloc[:, 1:])
# trainset.to_csv("./dataset/train_ADASYN_new_rotate.csv")
# trainset.to_csv("./dataset/train_ADASYN_new_withoutoversample.csv")
# validset.to_csv("./dataset/valid_ADASYN_new.csv")
# testset.to_csv("./dataset/test_ADASYN_new.csv")


