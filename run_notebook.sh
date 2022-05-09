
docker build -t notebook .

docker run -it -p 8888:8888 -v /$(pwd):/home/jovyan/work --rm --name jupyter notebook