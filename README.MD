use simple-faster-rcnn(Yun Chen) to detect multi human interaction
in UT_interact test dataset, the mAP is 60+
before use it,download VGG16 pretain_model put it into ./misc/checkpoints/
the dataset :UT_interact-- --frame --seq01
                          |       |
                          |        --seq0x...(0001.jpg,0002.jpg)
                          |
                          |
                           --ut_set --(test.txt,trainval.txt)
                          |
                          --frame --seq01
                          |       |
                          |        --seq0x...(0001.json,0002.json)
                          |
                           --video(20 videoes)
