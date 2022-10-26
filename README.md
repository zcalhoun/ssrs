# ssrs
Self-Supervised Remote Sensing Evaluation

This repository includes the code we used to run the experiments described in the paper "Self-supervised encoders are better transfer learners in remote sensing applications" (publication pending -- link to be added soon). The file ``swav_env.yml`` has the conda environment with all the dependecies required. We desribe below the step-by-step guide to run the following experiment: use pretrained ResNet model on ImageNet using SwAV, perform additional pretraining using the target dataset, test resulting model on test dataset and report pixel IoU.

1. Clone both the swav and ssrs repositories 
2. Under the swav repo, run ``main_swav.py`` using the appropriate script. See ``/swav/scripts/building/swav_800ep_pretrain_3000.sh`` for an example. Update the parameters as needed (DATASET_PATH, nproc_pernode, task)
3. Find the last checkpoint under your EXPERIMENT_PATH/checkpoints (``ckp-799.pth`` if training using 800 epochs)
4. Use ``save_model.sh`` to run  ``save_model.py`` with the right arguments (model_path, output_path) to save the model as a .pt file to load later for testing
5. Under the ssrs repo, run ``all_experiments.sh`` to run all the experiments described in the paper. You can change the ``DEVICE`` (CUDA=0) depending on your machine, and you can change the ``TASK`` you want to experiment with (solar, building, crop-delineation). To test using different encoders, you can change the ``--encoder`` parameter _AND ADD SUPPORT FOR THE NEWLY ADDED ENCODER_. This means:
    a. Adding the new encoder's name to the list of allowed choices for the encoder parameter under main.py
    b. Adding the new encoder under the ``load`` function of ``model/encoders.py``

6. Run ``BatchAnalysis.py`` to calculate IoU and pass to it the output_path of step 5 
7. Use the generated .csv file to visualize the IoU vs dataset size curves. You can use ``crop_delineation_final_results.ipynb`` as an example
