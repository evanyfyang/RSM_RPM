# bash/train_extractor.sh -d 14res

while getopts ':d:c:s:l:o:t:u:a:' opt
do
    case $opt in
        d)
        dataset="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
        s)
        seed="$OPTARG" ;;
        l)
        learning_rate="$OPTARG" ;;
        o)
        output_dir="$OPTARG" ;;
        t)
        use_prompt="$OPTARG" ;;
        u)
        use_super="$OPTARG" ;;
        a)
        table_loss_alpha="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ ! "${CUDA_IDS}" ]
then
    CUDA_IDS=0
fi


if [ ! "${seed}" ]
then
    seed=42
fi



if [ ! "${learning_rate}" ]
then
    learning_rate=15
fi



if [ ! "${output_dir}" ]
then
    output_dir="./output/t5"
fi



max_seq_length=-1
gradient_clip_val=1
warmup_steps=0
weight_decay=0.01

precision=32
train_batch_size=8
eval_batch_size=8
max_epochs=20

# change this to your t5-base path
model_name_or_path="t5-base"
data_dir="data/asqp_t5/"



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_supervising_quad.py \
  --accelerator=gpu \
  --devices=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --learning_rate ${learning_rate}e-5 \
  --train_batch_size ${train_batch_size} \
  --eval_batch_size ${eval_batch_size} \
  --seed ${seed} \
  --warmup_steps ${warmup_steps} \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length ${max_seq_length} \
  --output_sub_dir ${dataset} \
  --dataset ${dataset} \
  --max_epochs ${max_epochs} \
  --use_super ${use_super} \
  --use_prompt ${use_prompt} \
  --table_loss_alpha ${table_loss_alpha} \
  --do_train