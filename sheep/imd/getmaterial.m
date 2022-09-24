
for i=1:13
[tPic,~,tAlp]=imread(['card_',num2str(i),'.png']);
card(i).C=imresize(tPic,[145,130]);
card(i).A=imresize(tAlp,[145,130]);
end

store=imread('store.jpg');
bkg=imread('bkg.jpg');

[tPic,~,tAlp]=imread('re1.png');
re(1).C=tPic;
re(1).A=tAlp;
[tPic,~,tAlp]=imread('re2.png');
re(2).C=tPic;
re(2).A=tAlp;

author=char([20316 32773 58 32 115 108 97 110 100 97 114 101 114]);
gzh=char([20844 20247 21495 58 32 115 108 97 110 100 97 114 101 114 38543 31508]);

save material.mat store card bkg re author gzh