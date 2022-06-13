import subprocess

model_tag = "CIFAR100-N10-class--C3-5x16-bn_F-1024x1000x1000x1000_c100--i5000-lr0.0001-b256-pCvE-cifar10_4fc_1000hn_100epochs_128br_proper-fCvE--Hgen-Di2.0-VAE=C3-5x16-bnH_F-1024x1000x1000x1000_z100-GMM100pc-lrlf[0.0,0.2,0.8]--MSE"

params = model_tag.split("-")

imp_params = params[13]
fc_lay = 4
hidden_neurons = 1000
iters = 5000
epochs = 100
frequency = params[23][5:-1]
batch_replay = 256

results_dir_suffix = "frequency_fid"

subprocess.call(
    [
        "sh",
        "./scripts/fid_checks.sh",
        str(fc_lay),
        str(hidden_neurons),
        str(iters),
        str(epochs),
        str(frequency),
        str(batch_replay),
        str(model_tag),
        str(results_dir_suffix),
    ]
)
