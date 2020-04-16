# Description of Test Options

Let us take [`test_DIC_CelebA.json`](./train_DIC_CelebA.json) as an example. 

**Note**: Before you run `python test.py -opt options/test/*.json`, please carefully check options: `"scale"`, `"dataroot_HR"` and `"pretrained_path"` (if `"pretrain"` option is set to `"resume"` or `true`).

```c++
{
    "name": "CelebA" // test name
    "mode": "sr_align", // solver type: "sr_align" | "sr_align_gan"
    "degradation": "BI", // degradation model
    "gpu_ids": [0], // GPU ID to use
    "use_tb_logger": false, // whether to use tb_logger
    "scale": 8, // super resolution scale (*Please carefully check it*)
    "is_train": true, // whether train the model
    "rgb_range": 1, // maximum value of images
    "save_image": true, // whether saving visual results during training

    // test dataset specifications (you can place more than one test dataset here) (*Please carefully check dateset mode/root*)
    "datasets": { 
        // train dataset
        "test_CelebA": {
            "mode": "LRHR", // dataset mode: "HRLandmark" | "LRHR" | "LR"
            "name": "CelebA", // dataset name
            "dataroot_HR": "/home/jzy/datasets/SR_datasets/face_test/CelebA/HR", // HR dataset root
            "dataroot_LR": "/home/jzy/datasets/SR_datasets/face_test/CelebA/LR", // LR dataset root
            "data_type": "img", // data type: "img" (image files) | "npy" (binary files)
            "LR_size": 16, // input (LR) patch size
            "HR_size": 128, // input (HR) patch size
        }
    },

    // networks specifications
    "networks": {
        "which_model": "DIC", // network name
        "num_features": 48, // number of base feature maps
        "in_channels": 3, // number of input channels
        "out_channels": 3, // number of output channels
        "num_steps": 4, // number of time steps (T)
        "num_groups": 3 // number of projection groups (G)
        "detach_attention": false, // whether detach attention map to stop gradient flow
        "hg_num_feature": 256, // number of HourGlass features
        "hg_num_keypoints": 68, // number of face keypoints
        "num_fusion_block": 7 // number of blocks in attention fusion module
    },

    // solver specifications
    "solver": {
        "pretrained_path": "../experiments/DIC_in3f48_x8_DIC_CelebA/epochs/best_ckp.pth" // pre-trained model directory (for test)
    },
    "path": {
      "root": "../" // repository root
    }
}
```