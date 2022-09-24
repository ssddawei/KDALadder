import Storage from './storage-localfile.js';

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
    async register(groupName, groupCode, inviteCode) {
        await storage.createGroup(groupName, groupCode);
    }
    async login(groupCode) {

        throw "not imp"
    }
    async generateInviteCode() {

        throw "not imp"
    }
    async updateNameOrCode(oldGroupCode, groupName, groupCode) {
        await storage.updateGroup(oldGroupCode, groupCode ,groupName);
    }
    async saveMatch(groupCode, matchData, ladderData) {
        await storage.saveMatch(groupCode, matchData, ladderData);
    }
    async onClientConnect(clientID, ws) {
        
        throw "not imp"
    }
}


export default GroupController