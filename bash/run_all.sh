while getopts ':c:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done

bash bash/train_extractor_aste.sh -a 5 -d 14lap -l 30 -c ${CUDA_IDS} -s 40 -t 1 -u 1
bash bash/train_extractor_aste.sh -a 5 -d 14res -l 20 -c ${CUDA_IDS} -s 40 -t 1 -u 1
bash bash/train_extractor_aste.sh -a 5 -d 16res -l 20 -c ${CUDA_IDS} -s 40 -t 1 -u 1
bash bash/train_extractor_aste.sh -a 5 -d 15res -l 30 -c ${CUDA_IDS} -s 40 -t 1 -u 1
bash bash/train_extractor_quad.sh -a 1 -d rest15 -l 20 -c ${CUDA_IDS} -s 40 -t 1 -u 1
bash bash/train_extractor_quad.sh -a 1 -d rest16 -l 20 -c ${CUDA_IDS} -s 40 -t 1 -u 1
bash bash/train_extractor_acos.sh -a 1 -d rest16 -l 20 -c ${CUDA_IDS} -s 40 -t 1 -u 1
bash bash/train_extractor_acos.sh -a 1 -d laptop16 -l 20 -c ${CUDA_IDS} -s 40 -t 1 -u 1