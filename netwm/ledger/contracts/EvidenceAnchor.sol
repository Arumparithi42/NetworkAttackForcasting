// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title EvidenceAnchor
/// @notice Stores Merkle roots of ranges of forecast/alert records kept in the off-chain
///         evidence ledger. Only hashes are stored: no traffic, IP addresses or personal data.
contract EvidenceAnchor {
    event Anchored(bytes32 indexed root, uint256 fromSeq, uint256 toSeq, address indexed sensor);

    mapping(bytes32 => uint256) public anchoredAt;   // root => block timestamp

    function anchor(bytes32 root, uint256 fromSeq, uint256 toSeq) external {
        require(anchoredAt[root] == 0, "already anchored");
        anchoredAt[root] = block.timestamp;
        emit Anchored(root, fromSeq, toSeq, msg.sender);
    }
}
