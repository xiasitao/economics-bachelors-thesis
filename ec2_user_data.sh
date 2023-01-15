#! /bin/bash
cd /home/ec2-user
echo "aws s3 sync s3://econ-bachelors-thesis/ econ_bachelors_thesis" > s3_sync.sh
chmod +x s3_sync.sh
sh s3_sync.sh
chown -R ec2-user econ_bachelors_thesis

amazon-linux-extras install python3.8
python3.8 -m pip install pytask pandas==1.4.3 numpy transformers torch