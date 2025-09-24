// SPDX-License-Identifier: MIT

pragma solidity >=0.6.0 <0.9.0;

contract authorisationContract {

    //address owner;

    struct client{
        string wallet;
        bool exists;
    }
    
    mapping(string => client) public clientMap;

    function add(string memory _walletAddress) public {
        //require(msg.sender == owner, "You do not have the right privileges to do this");
        clientMap[_walletAddress] = client(_walletAddress,true);
    }

    function check(string memory _walletAddress) public view returns (bool) {
        if(clientMap[_walletAddress].exists){
            return true;
        } else {
            return false;
        }
    }

    function remove(string memory _walletAddress) public {
        //require(msg.sender == owner, "You do not have the right privileges to do this");
        clientMap[_walletAddress] = client(_walletAddress,true);
    }
}