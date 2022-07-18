# 客户端 webrtc 互联方案
基于对服务端最少依赖的构想，本方案不使用 stun server （代表无法穿越 NAT），no server（代表无需服务端计算资源，服务端无状态）

## Token Service 信令服务
webrtc 需要交换 sdp 来建立连接，本方案使用云存储(OSS)的方式实施。
- [OfferSDP] 发起端把 offersdp 写入 oss offer.sdp 文件
- [AnswerSDP] 接入端获取 offer.sdp 文件得到 offersdp，并且把 answersdp 写入 answer.sdp
- [Connect] 此时双方完成 sdp 交互，如果双方网络 UDP 互通，则可以建立 webrtc 连接
- [Clear] 连接建立后，发起端把 sdp 文件删除。发起端每次开启连接服务的时候，先清理 sdp 文件。
此方式，同一时间只支持一对发起/接入端。