
docker build -t notebook .

#docker run -p 8888:8888 --mount /Users/kzmq426/'OneDrive - AZCollaboration'/Desktop/Master-Thesis:/home/jovyan/work notebook

docker run -it -p 8888:8888 -v //c/Users/kzmq426/'OneDrive - AZCollaboration'/Desktop/Master-Thesis:/home/jovyan/work --rm --name jupyter notebook