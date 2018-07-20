gsutil cp gs://julius-ptl/imgs.tar ./
gsutil cp gs://julius-ptl/wm_resnet50_transfer.py ./
tar xvf ./imgs.tar
python3 wm_resnet50_transfer.py
gsutil cp ./wm_resnet50_result.csv gs://julius-ptl/
gsutil cp ./wm_resnet50_log.txt gs://julius-ptl/
gcloud compute instances stop gpu-k80 --zone=asia-east1-b --project=api-project-417210697275
