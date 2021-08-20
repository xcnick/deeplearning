- 配置Dockers HTTPS_PROXY

```bash
# 创建配置文件
sudo mkdir -p /etc/systemd/system/docker.service.d

sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf

# 加入以下文本
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:80"
Environment="HTTPS_PROXY=https://proxy.example.com:443"
Environment="NO_PROXY=localhost,127.0.0.1,docker-registry.example.com,.corp"

# 使修改生效
sudo systemctl daemon-reload
sudo systemctl restart docker

# 查看修改效果
sudo systemctl show --property=Environment docker
```

- 使用Docker挂载S3存储

```bash
# Docker run 指令需要增加
-cap-add SYS_ADMIN --device /dev/fuse --security-opt apparmor=unconfined
```

- 容器内挂载 S3

```bash
echo
AKIAIOSFODNN7MINIO:wJal3rXUt0nFEM6IbPx1Rf0iCY > .passwd-s3fs

chmod 600 .passwd-s3fs

mkdir s3mnt

s3fs test01bkt s3mnt -o
passwd_file=.passwd-s3fs -o url=https://oss.orientalmind.cn -o
use_path_request_style -o umask=0007,uid=1000,gid=1000
```

- 容器中的程序界面映射到主机中

```bash
docker run -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e QTX11NOMITSHM=1 -v $HOME/.Xauthority:/root/.Xauthority --device /dev/video0 lightpose:0.1 bash

# 可能需要在界面中设置
# 修改 DISPLAY 环境变量
export DISPLAY=:0

# 主机上修改权限，允许所有用户访问显示接口*
xhost +
```