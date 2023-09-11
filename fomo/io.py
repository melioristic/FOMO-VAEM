import h5py
import numpy as np

def read_benchmark_data(ds_path, pft="beech", xs_year_before = 3,  xd_year_from = 2,   xs_list=["age"], classify = False, target= 'MBR', normalize_xd = True, agg="monthly"):
    
    n_years = 3
    percentile = 90
    
    file_path = ds_path + f"train_{target}_{pft}_{agg}_{n_years}_years_10000ha.h5"
    Xd_train, Xs_train, Y_train, bins = _read_single_file(file_path=file_path, xs_year_before=xs_year_before, xd_year_from=xd_year_from, xs_list=xs_list)

    file_path = ds_path + f"val_{target}_{pft}_{agg}_{n_years}_years_10000ha.h5"
    Xd_val, Xs_val, Y_val, bins = _read_single_file(file_path=file_path, xs_year_before=xs_year_before, xd_year_from=xd_year_from, xs_list=xs_list)

    file_path = ds_path + f"test_{target}_{pft}_{agg}_{n_years}_years_10000ha.h5"
    Xd_test, Xs_test, Y_test, bins = _read_single_file(file_path=file_path, xs_year_before=xs_year_before, xd_year_from=xd_year_from, xs_list=xs_list)
    
    t = np.percentile(Y_train, percentile)

    if classify == True:
    
        Y_train = np.where(Y_train>t, 1, 0)
        Y_val = np.where(Y_val>t, 1, 0)
        Y_test = np.where(Y_test>t, 1, 0)

    
    if normalize_xd:
        mean_Xd = Xd_train.mean(axis = 0 )
        std_Xd = Xd_train.std(axis = 0)

        Xd_train = (Xd_train - mean_Xd)/std_Xd
        Xd_val = (Xd_val - mean_Xd)/std_Xd
        Xd_test = (Xd_test - mean_Xd)/std_Xd


    train = Xd_train, Xs_train[:, :,:], Y_train
    val = Xd_val, Xs_val[:, :,:], Y_val
    test = Xd_test, Xs_test[:, :,:], Y_test

    return {"train":train, "validation":val, "test":test, "bins":bins[:], "mean_Xd":mean_Xd, "std_Xd":std_Xd}


def _read_single_file(file_path, xs_year_before, xd_year_from, xs_list):

    steps_in_year = 365
    if "pentad" in file_path:
        steps_in_year = 73
    elif "monthly" in file_path:
        steps_in_year = 12
    names = ["age", "sv", "laicum", "h", "d"]

    assert 4>xs_year_before
    assert 3>xd_year_from>0

    index_list = []
    for each in xs_list:
        index_list.append(names.index(each))

    with h5py.File(file_path, "r") as f:

        start_sample = 3-xs_year_before
        end_sample = -start_sample if -start_sample!=0 else None
        start_time = (2-xd_year_from)*steps_in_year

        Xd = f["Xd"][:end_sample,start_time:, :]
        Xs = f["Xs"][start_sample:,:,index_list ]
        Y = f["Y"][:end_sample]
        bins = np.vstack(
            [
                f["age_bin_0"][:],
                f["sv_bin_1"][:],
                f["laitree_bin_2"][:],
                f["h_bin_3"][:],
                f["d_bin_4"][:],
            ]
        )[index_list]

    return Xd, Xs, Y, bins