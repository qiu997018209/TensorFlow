# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from app import app
from conf import *
from log import log

if __name__ == '__main__':
    args=get_args()
    log('\nhttp_host:{},http_port:{}'.format(args.http_host,args.http_port))
    app.run(debug=True, host=args.http_host, port=args.http_port)
