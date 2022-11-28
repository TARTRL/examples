export LIBCUDA_LOG_LEVEL=0
pip install xmltodict -i https://pypi.douban.com/simple/
pip install transformers==4.11.3 -i https://pypi.douban.com/simple/
pip install datasets==2.0.0 -i https://pypi.douban.com/simple/
pip install future -i https://pypi.douban.com/simple/
pip install consoleprinter -i https://pypi.douban.com/simple/
pip install seqeval -i https://pypi.douban.com/simple/
cp -r /TData/code/tasks /build
cp -r /TData/code/model /build
cp -r /TData/code/training /build
cp -r /TData/code/metrics /build
cp /TData/code/arguments.py /build