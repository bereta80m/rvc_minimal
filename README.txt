
python scripts\train.py `
  --dataset "C:\Users\josia\Documents\MyProjects\IA-Voice-Cloner\rvc_minimal_2\dataset_raw\badbunny" `
  --exp "badbunny_test" `
  --sr 40k `
  --version v2 `
  --if_f0 1 `
  --np 8 `
  --f0_method rmvpe_gpu `
  --gpus 0 `
  --gpus_rmvpe 0 `
  --pretrained_g "C:\Users\josia\Documents\MyProjects\IA-Voice-Cloner\rvc_minimal_2\assets\pretrained_v2\f0G40k.pth" `
  --pretrained_d "C:\Users\josia\Documents\MyProjects\IA-Voice-Cloner\rvc_minimal_2\assets\pretrained_v2\f0D40k.pth" `
  --batch_size 4 `
  --save_every_epoch 5 `
  --total_epoch 100 `
  --save_every_weights `
  --copy_to_models `
  --is_half 1






python scripts/convert.py `
   --input "C:\Users\josia\Documents\MyProjects\IA-Voice-Cloner\rvc_minimal_2\input\0LeadVocals_fixed.wav" `
   --output "output.wav" `
   --model "models\josias_test.pth" `
   --index "models\josias_test.index" `
   --index_rate 0.75 `
   --transpose 0 `
   --protect 0.33 `
   --rms_mix_rate 0.25







uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload




curl.exe -X POST "http://localhost:8000/api/train" `
>>   -H "Content-Type: application/json" `
>>   --data-binary "@train.json"
