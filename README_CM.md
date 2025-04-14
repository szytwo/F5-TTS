## 安装

```
conda create --prefix ./venv python==3.11
conda activate ./venv
conda install -c conda-forge montreal-forced-aligner

conda install -c conda-forge spacy sudachipy sudachidict-core

pip install -r ./api_requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install hanlp -U   # 安装或升级 hanlp 库

```

## 设置环境变量

MFA_ROOT_DIR 来存储全局配置设置和保存的模型

默认为 /home/docs/Documents/MFA

PKUSEG_HOME="/code/pkuseg"

## 下载字典与声音模型

```

mfa model download dictionary mandarin_china_mfa
mfa model download acoustic mandarin_mfa

```

官网 https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html
fastText 语言检测模型
https://fasttext.cc/docs/en/language-identification.html

## Docker镜像操作

```

docker build -t montreal-forced-aligner:1.0 . --progress=plain # 构建镜像
docker build -t montreal-forced-aligner:1.0 --cache-from montreal-forced-aligner:1.0 .
docker load -i montreal-forced-aligner-1.0.tar # 导入镜像
docker save -o montreal-forced-aligner-1.0.tar montreal-forced-aligner:1.0 # 导出镜像
docker-compose up -d # 后台运行容器
docker builder prune -a #强制清理所有构建缓存

```

## 安全压缩 WSL2/Docker 虚拟磁盘

```

wsl --shutdown # 关闭Docker/WSL
diskpart # 进入磁盘管理工具
select vdisk file="D:\Docker\DockerDesktopWSL\disk\docker_data.vhdx" # 选择虚拟磁盘文件（即 Docker 的 WSL2 数据文件
attach vdisk readonly # 以只读模式挂载磁盘
compact vdisk # 压缩虚拟磁盘文件
detach vdisk # 卸载磁盘
exit # 退出 diskpart

```

## GIT

```
git pull # 拉取
git push # 推送

git submodule add https://github.com/szytwo/fastText.git # 添加子模块
git submodule update --init --recursive # 初始化子模块

git branch -r # 查看分支
git branch -m main # 重命名分支
git branch --set-upstream-to=origin/main main #关联远程分支origin/main 

git remote -v # 查看远程仓库
git remote remove origin # 移除远程仓库连接，origin，upstream

# 添加新的远程仓库，origin，upstream
git remote add upstream https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git 

git fetch upstream # 从远程仓库拉取更新，origin，upstream
git checkout main # 切换到主分支
git merge upstream/main # 合并到本地分支,主分支名称可能是 ，origin，upstream，master,main 

```