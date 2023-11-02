
## ADEPT
# Loci Looped on adept
python -m scripts.exec.eval -cfg out/pretrained/adept/loci_looped/cfg_loci_looped_run4.json -load out/pretrained/adept/loci_looped/net1_loci_looped.pt -n 2
python -m scripts.exec.eval -cfg out/pretrained/adept/loci_looped/cfg_loci_looped_run4.json -load out/pretrained/adept/loci_looped/net2_loci_looped.pt -n 2
python -m scripts.exec.eval -cfg out/pretrained/adept/loci_looped/cfg_loci_looped_run4.json -load out/pretrained/adept/loci_looped/net3_loci_looped.pt -n 2

# Loci Unlooped on adept
python -m scripts.exec.eval -cfg out/pretrained/adept/loci_unlooped/cfg_loci_unlooped_run3.json -load out/pretrained/adept/loci_unlooped/net1_loci_unlooped.pt -n 2

# SAVI on adept
python -m scripts.exec.eval_savi -load out/pretrained/adept/savi/savi_masks_adept.pkl -n 2

## CLEVRER
# Loci Looped on clevrer
python -m scripts.exec.eval -cfg out/pretrained/clevrer/loci_looped/cfg_loci_looped_run3.json -load out/pretrained/clevrer/loci_looped/net_loci_looped.pt -n 2

# Loci Unlooped on clevrer
python -m scripts.exec.eval -cfg out/pretrained/clevrer/loci_unlooped/cfg_loci_unlooped_run3.json -load out/pretrained/clevrer/loci_unlooped/net_loci_unlooped.pt -n 2

