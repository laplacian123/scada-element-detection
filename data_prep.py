import os
import argparse
import numpy as np
import shutil
import yaml

GEN_IMG_DIR = "./gen_img"
DATA_BASE_DIR = "./datasets/dataset"
RAND_SEED = 87
SAMPLE_FRAC = 0.75

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img", help = "image base directory", default = GEN_IMG_DIR)
    parser.add_argument("-d", "--distination", help = "data distination base directory", default = DATA_BASE_DIR)
    parser.add_argument("-r", "--randomseed", help = "random seed", default = RAND_SEED)
    parser.add_argument("-s", "--samplingfrac", help = "fraction of sampling", default = SAMPLE_FRAC)

    args = parser.parse_args()

    img_base_dir = args.img
    dist_base_dir = args.distination
    random_seed = int(args.randomseed)
    sample_frac = float(args.samplingfrac)

    cls_list = os.listdir(img_base_dir)

    if not os.path.exists(dist_base_dir):
        os.makedirs(dist_base_dir)

    img_data_dir = os.path.join(dist_base_dir, "images")
    annot_data_dir = os.path.join(dist_base_dir, "labels")
    
    if not os.path.exists(img_data_dir):
        os.mkdir(img_data_dir)
    
    if not os.path.exists(annot_data_dir):
        os.mkdir(annot_data_dir)

    
    for i in range(len(cls_list)):
        rng = np.random.default_rng(seed = random_seed + i)
        cls_dir = os.path.join(img_base_dir, cls_list[i])
        fname_list = [s[:-4] for s in os.listdir(cls_dir)]
        n = len(fname_list)
        sample_flags = [True if flag < sample_frac else False for flag in rng.uniform(0.0, 1.0, size = n)]

        if cls_list[i] == "background":
            for j in range(n):
                if sample_flags[i]: 
                    shutil.copy(os.path.join(cls_dir, fname_list[j] + ".jpg"), img_data_dir)
                    
                else:
                    break

        else:
            for j in range(n):   
                if sample_flags[i]: 
                    shutil.copy(os.path.join(cls_dir, fname_list[j] + ".jpg"), img_data_dir)
                    f = open(os.path.join(annot_data_dir, f"{fname_list[j]}.txt"), "w")
                    f.writelines(f"{i} 0.500000 0.500000 1.000000 1.000000")
                    f.close()
                else:
                    break

    config = {
        "path": "." + dist_base_dir,  # dataset root dir
        "train": "images",  
        "val": "images",  
        "names": {}
    }


    names = {i: cls_list[i] for i in range(len(cls_list))}
    config["names"] = names

    f = open('dataset.yaml', 'w')
    yaml.dump(config, f)
    f.close()
