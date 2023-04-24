DIR=venv

if [ -d "$DIR" ];
then
    echo "$DIR directory exists."
    source venv/bin/activate
    python camoscio.py
else
	echo "$DIR directory does not exist."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r https://raw.githubusercontent.com/teelinsan/camoscio/main/requirements.txt
    # get python version
    PYTHON_VERSION=$(python --version | cut -c8-11)
    # Fix bitsandbytes library
    cp venv/lib/python$PYTHON_VERSION/site-packages/bitsandbytes/libbitsandbytes_cuda120.so venv/lib/python$PYTHON_VERSION/site-packages/bitsandbytes/libbitsandbytes_cpu.so
    python camoscio.py
fi