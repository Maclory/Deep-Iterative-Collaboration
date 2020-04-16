import argparse, time, os, json
from collections import OrderedDict
import imageio

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Test Super Resolution Models')
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()

    # create folders
    util.mkdir_and_rename(opt['path']['res_root'])
    option.save(opt)

    # create test dataloader
    bm_names = []
    test_loaders = []
    for ds_name, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print('===> Test Dataset: [%s]   Number of images: [%d]' %
              (dataset_opt['name'], len(test_set)))
        bm_names.append(dataset_opt['name'])

    # create solver (and load model)
    solver = create_solver(opt)
    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s" % (model_name, scale,
                                                          degrad))

    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [%s]" % bm)

        sr_list = []
        path_list = []

        total_psnr = []
        total_ssim = []
        total_time = []
        res_dict = OrderedDict()

        need_HR = False if test_loader.dataset.__class__.__name__.find(
            'HR') < 0 else True

        for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            solver.feed_data(batch, need_HR=need_HR, need_landmark=False)

            # calculate forward time
            t0 = time.time()
            solver.test()
            t1 = time.time()
            total_time.append((t1 - t0))

            visuals = solver.get_current_visual(need_HR=need_HR)
            sr_list.append(visuals['SR'][-1])

            # calculate PSNR/SSIM metrics on Python
            if need_HR:
                psnr, ssim = util.calc_metrics(
                    visuals['SR'][-1], visuals['HR'], crop_border=scale)
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                path_list.append(
                    os.path.basename(batch['HR_path'][0]).replace(
                        'HR', model_name))
                # print(
                #     "[%d/%d] %s || PSNR(dB)/SSIM: %.2f/%.4f || Timer: %.4f sec ."
                #     % (iter + 1, len(test_loader),
                #        os.path.basename(batch['HR_path'][0]), psnr, ssim,
                #        (t1 - t0)))
                res_dict[path_list[-1]] = {'psnr': psnr, 'ssim': ssim, 'time': t1 - t0}

            else:
                path_list.append(os.path.basename(batch['LR_path'][0]))
                # print("[%d/%d] %s || Timer: %.4f sec ." %
                #       (iter + 1, len(test_loader),
                #        os.path.basename(batch['LR_path'][0]), (t1 - t0)))

        if need_HR:
            print("---- Average PSNR(dB) /SSIM /Speed(s) for [%s] ----" % bm)
            average_res_str = "PSNR: %.2f      SSIM: %.4f      Speed: %.4f" % \
                  (sum(total_psnr) / len(total_psnr), sum(total_ssim) /
                   len(total_ssim), sum(total_time) / len(total_time))
            print(average_res_str)
        else:
            print("---- Average Speed(s) for [%s] is %.4f sec ----" %
                  (bm, sum(total_time) / len(total_time)))

        # save SR results for further evaluation on MATLAB
        save_img_path = os.path.join(opt['path']['res_root'], bm)

        print("===> Saving SR images of [%s]... Save Path: [%s]\n" %
              (bm, save_img_path))

        if not os.path.exists(save_img_path): os.makedirs(save_img_path)
        for img, name in zip(sr_list, path_list):
            imageio.imwrite(os.path.join(save_img_path, name), img)
        if need_HR:
            with open(os.path.join(save_img_path, 'result.json'), 'w') as f:
                json.dump(res_dict, f, indent=2)
            with open(os.path.join(save_img_path, 'average_result.txt'), 'w') as f:
                f.write(average_res_str + '\n')

    print("==================================================")
    print("===> Finished !")


if __name__ == '__main__':
    main()
