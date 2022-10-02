import Koa from 'koa'
// const route = require('koa-route');
import Router from '@koa/router'
import Websocket from 'koa-websocket'
import KoaBody from 'koa-body'
import Cors from '@koa/cors'
import StaticServe from 'koa-static'
import Conditional from 'koa-conditional-get'
import Etag from 'koa-etag'

import GroupController from './app.js'


const app = Websocket(new Koa());
const router = new Router();
const wsRouter = new Router();

const groupCtrl = new GroupController();

wsRouter.get('/', async (ctx, next) => {
    // `ctx` is the regular koa context created from the `ws` onConnection `socket.upgradeReq` object.
    // the websocket is added to the context on `ctx.websocket`.

    let userAgent = ctx.request.req.headers["user-agent"]
    let ip = ctx.ip
    groupCtrl.onClientConnect(ctx.websocket, userAgent, ip);
    
    return next;
});

router.use(KoaBody())

function apiDoneWithEmpty(ctx) {
    ctx.status = 200;
    ctx.body = ""
}

// 注册团体
router.post("/v1/group/new", async (ctx, next) => {
    let groupData = ctx.request.body;

    ctx.body = {
        groupCodeHashPath: await groupCtrl.register(groupData.groupName, groupData.groupCode, groupData.inviteCode)
    }
    
})

// 验证 groupCode，返回目录名
router.post("/v1/group/hash", async (ctx, next) => {
    let groupData = ctx.request.body;

    ctx.body = {
        groupCodeHashPath: await groupCtrl.login(groupData.groupCode)
    }
})

// 更新 group 信息
router.post("/v1/group/update", async (ctx, next) => {
    let groupData = ctx.request.body;

    await groupCtrl.updateNameOrCode(groupData.groupCode, groupData.groupName, groupData.newGroupCode);
})

router.post("/v1/group/info", async (ctx, next) => {
    let groupData = ctx.request.body;

    ctx.body = await groupCtrl.getInfo(groupData.groupCode);
})

// 保存赛局
router.post("/v1/group/match", async (ctx, next) => {
    let data = ctx.request.body;

    await groupCtrl.saveMatch(data.groupCode, data.matchData, data.ladderData);
})

// import fs from 'fs'
// router.get("/", async (ctx) => {
//     ctx.body = fs.readFileSync("./index.html")
//     ctx.set("Content-Type", "text/html")
// })

// Error handling
app.use(async (ctx, next) => {
    try {
        await next();
        if(ctx.status != 304 && !ctx.body) {
            apiDoneWithEmpty(ctx);
        }
    } catch (err) {
        console.log("onError", err)
        // will only respond with JSON
        ctx.status = err.statusCode || err.status || 500;
        ctx.body = {
            message: err.message || err
        };
    }
})

app.use(Cors());
app.use(Conditional())
app.use(Etag())
app.use(StaticServe("data"))
app.use(StaticServe("../"))
app.use(router.routes()).use(router.allowedMethods());
app.ws.use(wsRouter.routes()).use(wsRouter.allowedMethods());

app.listen(8080);