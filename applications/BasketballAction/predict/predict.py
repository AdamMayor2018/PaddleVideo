
import os
import sys
import json
import time


sys.path.append('action_detect')
from applications.BasketballAction.predict.action_detect.action  import ActionDetection

if __name__ == '__main__':
    start = time.time()
    dataset_dir = "/data/cx/ysp-2024/PaddleVideo/applications/BasketballAction/predict/datasets/NBA/"
    
    model_predict = ActionDetection(cfg_file="configs_basketball/configs_basketball.yaml")
    model_predict.load_model()
    
    video_url = os.path.join(dataset_dir, 'mp4.list')
    with open(video_url, 'r') as f:
        lines = f.readlines()
    lines = [os.path.join(dataset_dir, "mp4", os.path.basename(k.strip())) for k in lines]
    
    results = []
    for line in lines:
        video_name = line
        print(video_name)

        imgs_path = video_name.replace(".mp4", "").replace("mp4", "frames")
        pcm_path = video_name.replace(".mp4", ".pcm").replace("mp4", "pcm")

        bmn_results, action_results = model_predict.infer(imgs_path, pcm_path)
        results.append({'video_name': line,
                        'bmn_results': bmn_results, 
                        'action_results': action_results})

    end = time.time()
    print(f"process time: {end -start}s")

    with open('results.json', 'w', encoding='utf-8') as f:
       data = json.dumps(results, indent=4, ensure_ascii=False)
       f.write(data)

