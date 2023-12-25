cd emlp-pytorch || exit

python3 particle_interaction.py --model scalar_mlp --sample_size 1 --eval_sample_size 1 --seed 42
python3 particle_interaction.py --model scalar_mlp --sample_size 1 --eval_sample_size 1 --seed 123
python3 particle_interaction.py --model scalar_mlp --sample_size 1 --eval_sample_size 1 --seed 321

cd .. || exit
