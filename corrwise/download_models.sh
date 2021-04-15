#!/bin/bash
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip

cp models/raft-things.pth flow_models/raft/

rm -r models.zip models
