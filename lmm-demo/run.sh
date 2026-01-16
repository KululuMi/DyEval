export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# STREAMLIT_SERVER_HEADLESS=true python -m streamlit run home.py  --server.fileWatcherType none --server.address=127.0.0.1 --server.address=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true python -m streamlit run home.py  --server.fileWatcherType none --server.address=127.0.0.1 #--server.address=0.0.0.0