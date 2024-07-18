FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
RUN apt-get update

# installing graphviz is a pain in the ass
RUN apt-get install -y graphviz python3-pydot libgraphviz-dev  python3-pygraphviz

# install requirements
ADD requirements.txt /tmp
ADD packages_install.cmd /tmp
RUN bash /tmp/packages_install.cmd

COPY . .

WORKDIR /app
