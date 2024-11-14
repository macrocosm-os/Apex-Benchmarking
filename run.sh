#!/bin/bash

# Run the install script
bash scripts/install.sh

# Check if the install script was successful
if [ $? -eq 0 ]; then
    echo "Installation successful. Proceeding to run benchmark.py"
    
    # Run the benchmark script with Poetry
    poetry run pm2 start "python benchmark.py" --name sn1-benchmark

    pm2 log sn1-benchmark

    # Check if the benchmark script ran successfully
    if [ $? -eq 0 ]; then
        echo "Benchmark completed successfully."
    else
        echo "Benchmark failed to run."
        exit 2
    fi
else
    echo "Installation failed. Aborting."
    exit 1
fi