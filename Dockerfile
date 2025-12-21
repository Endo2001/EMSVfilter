FROM rockylinux:9.3.20231119

RUN dnf install -y python3 python3-pip wget && dnf clean all && rm -rf /var/cache/dnf
RUN pip3 install torch torchvision timm pillow matplotlib numpy
RUN wget -O /usr/local/bin/best_dml_all.pth https://github.com/Endo2001/EMSVfilter/releases/download/0.1/best_dml_all.pth
ADD EMSVfilter.py diffpair_dataset.py eval_diff_cam.py eval_threshold_search.py /usr/local/bin
