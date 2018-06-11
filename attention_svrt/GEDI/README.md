Data is prepared according to tensorflow's build_image_data routines. This puts our images into tfrecords.

0: Set the gedi_config.py file to point to your correct paths and dataset

-Then either-
1: Produce image patches from gedi tiffs with: python extract_frames_from_GEDI.py 
2: Package the images into a format for efficient learning: python prepare_tf_records.py

-Or run-
1/2: sh run_preprocessing_scripts.sh (make sure to point to an appropriate gpu)

-Then-
3: Train your model on the data: python train_vgg16.py
4: Test your model on the data: python inference_vgg16.py (not yet created)

-Or run-
3/4: sh train_test_models.sh


TODO:

1. Add data augmentation into the tfqueue pipeline and greatly increase the training iterations.
2. Merge with the Heatmaps experiment
3. Incorporate shared variables during in train_vgg16 so that we can get validation accuracy during training