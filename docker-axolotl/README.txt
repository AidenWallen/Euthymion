// Build container
docker build -t axolotl-image .

// Run container
docker run -it --gpus all --rm -v "C:\Users\aiden\Euthymion\docker-axolotl:/app" -w /app --name axolotl-container axolotl-image


// Run training config
axolotl train config.yml

// Run test Euthymion
python test_euthy.py