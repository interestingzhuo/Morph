
import os
import shutil
with open("/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/albertzli/code/mf/datasets/physics_optimized_filename_to_test_smpl_pkl_filename.txt") as f:
    data = f.readlines()

os.makedirs("sample_fbx_out_generated", exist_ok=True)
os.makedirs("sample_fbx_out", exist_ok=True)

from tqdm import tqdm
for line in tqdm(data[:100]):
    gen_id, fix_id = line.strip().split("####")
    try:
        shutil.copy(f"/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/albertzli/code/mf/datasets/fbx/momask_test/{gen_id}.fbx", f"/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/albertzli/code/mf/datasets/fbx/zx/{gen_id}.fbx")
        shutil.copy(f"/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/albertzli/code/mf/datasets/fbx/test_smpl_pkl_to_mf_motion/{fix_id}.fbx", f"/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/albertzli/code/mf/datasets/fbx/zx/{fix_id}.fbx")
    except:
        continue

    
   # 001190####motions_amass_test_5518####a person walks from back to front, hops on right leg and then stops.