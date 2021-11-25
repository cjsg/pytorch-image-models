python avvalidate.py /tmp/imgnet/ --dataset imgnet --checkpoint output/train/2021-11-16-mr-mini-bottomup/last.pth.tar --model mr_mini_bottomup --batch-size 40 --num-gpu 8 --attacks fgm --attack-sizes 0. .0001 .00018 .00032 .00056 .001 .0018 .0032 .0056 .01 .017 .032 .056 .1 .17 .32 .56 > output/train/2021-11-16-mr-mini-bottomup/validation.out

python avvalidate.py /tmp/imgnet/ --dataset imgnet --checkpoint output/train/2021-11-16-mr-mini-bottomup-wconvs/last.pth.tar --model mr_mini_bottomup_wconvs --batch-size 40 --num-gpu 8 --attacks fgm --attack-sizes 0. .0001 .00018 .00032 .00056 .001 .0018 .0032 .0056 .01 .017 .032 .056 .1 .17 .32 .56 > output/train/2021-11-16-mr-mini-bottomup-wconvs/validation.out

python avvalidate.py /tmp/imgnet/ --dataset imgnet --checkpoint output/train/2021-11-16-mr-mini/last.pth.tar --model mr_mini --batch-size 40 --num-gpu 8 --attacks fgm --attack-sizes 0. .0001 .00018 .00032 .00056 .001 .0018 .0032 .0056 .01 .017 .032 .056 .1 .17 .32 .56 > output/train/2021-11-16-mr-mini/validation.out

python avvalidate.py /tmp/imgnet/ --dataset imgnet --checkpoint output/train/2021-11-16-nest-mini-modified/last.pth.tar --model nest_mini_modified --batch-size 40 --num-gpu 8 --attacks fgm --attack-sizes 0. .0001 .00018 .00032 .00056 .001 .0018 .0032 .0056 .01 .017 .032 .056 .1 .17 .32 .56 > output/train/2021-11-16-nest-mini-modified/validation.out

python avvalidate.py /tmp/imgnet/ --dataset imgnet --checkpoint output/train/2021-11-16-nest-mini/last.pth.tar --model nest_mini --batch-size 40 --num-gpu 8 --attacks fgm --attack-sizes 0. .0001 .00018 .00032 .00056 .001 .0018 .0032 .0056 .01 .017 .032 .056 .1 .17 .32 .56 > output/train/2021-11-16-nest-mini/validation.out
