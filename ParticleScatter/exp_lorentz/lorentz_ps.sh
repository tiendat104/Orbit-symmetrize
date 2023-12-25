cd emlp-pytorch || exit

python3 particle_interaction.py --seed 42
python3 particle_interaction.py --seed 123
python3 particle_interaction.py --seed 321

cd .. || exit
