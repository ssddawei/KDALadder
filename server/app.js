import Storage from './storage-localfile.js';
import path from 'path';
import md5 from 'md5';

let storage = new Storage();

class GroupClient {
    constructor(id, name) {
        this.clientID = id
        this.clientName = name
    }
}

class GroupContext {
    constructor(groupName) {
        this.groupName = groupName;
        this.clients = {}
    }
}

class GroupController {
    constructor() {
        /*
            {
                [groupCodeHashPath]: {
                    [md5(userAgent)]: ws,
                    subgroups: {
                        [name]: [clientID]
                    }
                }
            }
        */
        this.wsClients = {} // all websocket clients
    }
    async register(groupName, groupCode, inviteCode) {
        return await storage.createGroup(groupName, groupCode);
    }
    async login(groupCode) {
        let groupCodeHashPath = await storage.findPath(groupCode);
        return path.basename(groupCodeHashPath)
    }
    async generateInviteCode() {

        throw "not imp"
    }
    async updateNameOrCode(oldGroupCode, groupName, groupCode) {
        await storage.updateGroup(oldGroupCode, groupCode ,groupName);
    }
    async getInfo(groupCode) {
        return await storage.getGroup(groupCode);
    }
    async getToken(groupCode) {
        let groupCodeHashName = path.basename(await storage.findPath(groupCode));
        return await storage.groupCodeToken(groupCodeHashName);
    }
    async verifyToken(token) {
        return await storage.verifyGroupCodeToken(token);
    }
    async saveMatch(groupCode, matchData, ladderData) {
        await storage.saveMatch(groupCode, matchData, ladderData);
    }
    async saveMember(groupCode, member) {
        await storage.saveMember(groupCode, member)
    }
    async updateMember(groupCode, member) {
        await storage.updateMember(groupCode, member)
    }
    async onClientConnect(ws, userAgent, ip) {
        ws.once('message', async (msg)=>{
            console.log(msg)
            let groupCodeHashPath;
            if(msg && msg.length > 16) {
                // > 16 is token
                if(await storage.verifyGroupCodeToken(msg.toString()).catch(err => false)) {
                    let groupCodeHash = msg.toString().split(".")[0]; 
                    groupCodeHashPath = await storage.hasPath(groupCodeHash).catch(err => {
                        ws.close(1000, "group not exist")
                    })
                } else {
                    ws.close(1000, "token invalid")
                }
            }
            else {
                // < 16 is groupCode
                groupCodeHashPath = await storage.findPath(msg.toString()).catch(err => {
                    ws.close(1000, "groupCode not exist")
                })
            }

            if(!groupCodeHashPath) return;

            this.wsClients[groupCodeHashPath] = this.wsClients[groupCodeHashPath] || {subgroups:{}}

            let groupClients = this.wsClients[groupCodeHashPath];

            // use md5 as clientID
            let clientID = md5(userAgent + ip);

            // tailing "-1" when existed
            while(groupClients[clientID]) {
                if(!__idx) { var __idx = 1 }
                clientID = md5(userAgent + ip) + `-${__idx++}`
            }

            ws.send(clientID)

            groupClients[clientID] = ws;

            // clean up
            ws.once('close', ()=>{
                delete groupClients[clientID]
                Object.entries(groupClients.subgroups).forEach(([idx, i])=>{
                    i.splice(i.findIndex(el=>el==clientID), 1) // remove self
                    if(i.length == 0) { // delete when empty
                        delete groupClients.subgroups[idx] 
                    }
                })
            })

            ws.on('message', (msg)=>{
                let targetClients = Object.entries(this.wsClients[groupCodeHashPath])
                    .filter(i=>i[0]!=clientID)
                    .filter(i=>i[0]!="subgroups");

                msg = JSON.parse(msg);

                console.log(msg)
                // msg only send to subgroup clients
                if(msg.subgroup) {
                    groupClients.subgroups[msg.subgroup] = groupClients.subgroups[msg.subgroup] || [];
                    let subgroup = groupClients.subgroups[msg.subgroup];
                    let allsubgroups = Object.values(groupClients.subgroups).flatMap(i=>i);

                    // add to subgroup
                    if(!subgroup.find(i=>i==clientID)) {
                        subgroup.push(clientID);
                    }

                    // filter out other clients then current subgroup
                    targetClients = targetClients.filter(([cid, clientWS]) => {
                        return subgroup.find(i=>i==cid) || !allsubgroups.find(i=>i==cid)
                    })
                }
                this.boradcastClientMessage(targetClients.map(i=>i[1]), clientID, msg);
            })
        })
    }
    async boradcastClientMessage(targetClients, fromCid, msg) {
        // console.log("client message", clientID, msg)
        if(msg.sync == undefined || msg.sync) {
            // send to all groupClients without self
            targetClients.map(clientWS => {
                clientWS.send(JSON.stringify(msg));
            })
        }
    }
}


export default GroupController