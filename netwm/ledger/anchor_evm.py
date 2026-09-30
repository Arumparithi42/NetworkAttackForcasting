"""OPTIONAL: anchor ledger Merkle roots on a LOCAL Ethereum-compatible chain (Hardhat / Anvil).

    npx hardhat node          # or: anvil
    python scripts/verify_ledger.py artifacts/ledger.sqlite --anchor http://127.0.0.1:8545

Requires `pip install web3 py-solc-x` (not part of the core install). The core system never
depends on this module.
"""
from __future__ import annotations

from pathlib import Path

CONTRACT = Path(__file__).with_name("contracts") / "EvidenceAnchor.sol"


def anchor_root(ledger, rpc_url: str = "http://127.0.0.1:8545", from_seq: int = 1,
                to_seq: int | None = None) -> str:
    try:
        from solcx import compile_source, install_solc
        from web3 import Web3
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("EVM anchoring needs: pip install web3 py-solc-x") from exc
    to_seq = to_seq or ledger.head()[0]
    root = ledger.merkle(from_seq, to_seq)
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    acct = w3.eth.accounts[0]                         # unlocked dev account of the local node
    install_solc("0.8.20")
    compiled = compile_source(CONTRACT.read_text(), output_values=["abi", "bin"], solc_version="0.8.20")
    _, iface = compiled.popitem()
    Anchor = w3.eth.contract(abi=iface["abi"], bytecode=iface["bin"])
    tx = Anchor.constructor().transact({"from": acct})
    address = w3.eth.wait_for_transaction_receipt(tx).contractAddress
    c = w3.eth.contract(address=address, abi=iface["abi"])
    tx = c.functions.anchor(bytes.fromhex(root), from_seq, to_seq).transact({"from": acct})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    ledger.record_anchor(from_seq, to_seq, root, f"{rpc_url}#{address}", receipt.transactionHash.hex())
    return receipt.transactionHash.hex()
