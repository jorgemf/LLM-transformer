#!/bin/bash
systemctl stop docker.socket
systemctl stop docker.service
systemctl stop containerd.service
socat UNIX-LISTEN:/var/run/docker.sock,fork,reuseaddr,unlink-early,group=docker,mode=777 TCP:ishtar:4243