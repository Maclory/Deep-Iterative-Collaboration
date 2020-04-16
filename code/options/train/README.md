# Description of Train Options

Let us take [`train_DIC_CelebA.json`](./train_DIC_CelebA.json) as an example. 

**Note**: Before you run `python train.py -opt options/train/*.json`, please carefully check options: `"scale"`, `"dataroot_HR"` and `"pretrained_path"` (if `"pretrain"` option is set to `"resume"` or `true`).

```c++
{
    "name": "DIC_CelebA" // experiment name
    "mode": "sr_align", // solver type: "sr_align" | "sr_align_gan"
    "gpu_ids": [0, 1], // GPU ID to use
    "use_tb_logger": true, // whether to use tb_logger
    "scale": 8, // super resolution scale (*Please carefully check it*)
    "is_train": true, // whether train the model
    "rgb_range": 1, // maximum value of images
    "save_image": true, // whether saving visual results during training
    // dataset specifications including "train" dataset (for training) and "val" dataset (for validation) (*Please carefully check dateset mode/root*)
    "datasets": { 
        // train dataset
        "train": { 
            "mode": "HRLandmark", // dataset mode: "HRLandmark" | "LRHR" | "LR", "HRLandmark" is required during training
            "name": "CelebALandmarkTrain", // dataset name
            "dataroot_HR": "/home/jzy/datasets/SR_datasets/CelebA/img_celeba", // HR dataset root
            "info_path": "/home/jzy/datasets/SR_datasets/CelebA/new_train_info_list.pkl", // path to landmark annotations
            "data_type": "img", // data type: "img" (image files) | "npy" (binary files)
            "n_workers": 8, // number of threads for data loading
            "batch_size": 8, // input batch size
            "LR_size": 16, // input (LR) patch size
            "HR_size": 128, // input (HR) patch size
            // data augmentation
            "use_flip": false, // whether use horizontal and vertical flips
            "use_rot": false, // whether rotate 90 degrees
            "sigma": 1, // sigma of heatmap
        },
        // validation dataset, small as train set
        "val": {
            "mode": "HRLandmark",
            "name": "CelebALandmarkVal",
            "dataroot_HR": "/home/jzy/datasets/SR_datasets/CelebA/img_celeba",
            "info_path": "/home/jzy/datasets/SR_datasets/CelebA/new_val_info_list.pkl",
            "data_type": "img",
            "LR_size": 16,
            "HR_size": 128,
            "sigma": 1
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
        "type": "ADAM", // optimizer to use (only "ADAM" is provided)
        "learning_rate": 1e-4, // learning rate
        "weight_decay": 0, // weight decay
        "lr_scheme": "MultiStepLR", // learning rate scheduler (only "MultiStepLR" is provided)
        "lr_steps": [1e4, 2e4, 4e4, 8e4], // milestone for learning rate scheduler
        "lr_gamma": 0.5, // gamma for learning rate scheduler
        "manual_seed": 0, // manual seed for random & torch.random
        "save_freq": 5e3, // how many iterations to wait before saving models
        "val_freq": 2e3, // how many iterations to wait before validation
        "niter": 1.5e5, // total number of iterations
        "num_save_image": 20, // number of images to save during validation
        "log_full_step": true, // whether to save&log SR images of all the intermediate steps.
        "pretrain": false, // pre-train mode: false (from scratch) | "resume" (resume from specific checkpoint) | true (finetune a new model based on a specific model)
        // "pretrained_path": '../model/DIC_CelebA.pth', // path to pretrained model (if "pretrain" is not false)
        "release_HG_grad_step": 2e3, // which iteration to start optimizing HourGlass parameters
        "HG_pretrained_path": "../models/FB_HG_68_CelebA.pth", // where to load pretrained HourGlass model
        // loss settings
        "loss": {
            // pixel loss
            "pixel": {
                "loss_type": "l1", // loss type: "l1" | "l2"
                "weight": 1, // loss weight
            },
            // landmark detection loss
            "align": {
                "loss_type": "l2", // loss type: "l1" | "l2"
                "weight": 1e-1, // loss weight
            }
        }
    },
    "logger": { // logger setting
        "print_freq": 250 // how many iterations to wait before printing current training information
    },
    "path": {
      "root": "../" // repository root
    }
}
```
