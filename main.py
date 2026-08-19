from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import unicodedata
from email.utils import parsedate_to_datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from orchestator import (
	generate_session_id,
	get_model,
	get_session_summary,
	get_trace_info,
	print_results,
	run_llm_call,
)


DEFAULT_DATA_ROOT = Path("data") #Default Data Root
DEFAULT_OUTPUT_ROOT = Path("outputs")
MAX_RECENT_TX_CONTEXT = 8
MAX_COMMS_CONTEXT = 6
ROUTER_CHUNK_SIZE = int(os.getenv("ROUTER_CHUNK_SIZE", "120"))
CANDIDATE_RATIO = float(os.getenv("CANDIDATE_RATIO", "0.12"))
MIN_CANDIDATES = int(os.getenv("MIN_CANDIDATES", "14"))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "60"))
MAX_WORKERS = int(os.getenv("FRAUD_AGENT_MAX_WORKERS", "6"))
BORDERLINE_REVIEW_TOPK = int(os.getenv("BORDERLINE_REVIEW_TOPK", "4"))

SUSPICIOUS_TEXT_RE = re.compile(
	r"paypa1|amaz0n|account\s*lock|suspicious\s*(login|sign[- ]?in)|verify\s+(your\s+)?(account|identity|payment)|immediate\s*action\s*required|unusual\s+login",
	flags=re.IGNORECASE,
)


@dataclass
class AgentVote:
	agent: str
	risk_score: float
	vote: str
	confidence: float
	reasons: List[str]


@dataclass
class Decision:
	transaction_id: str
	amount: float
	is_fraud: bool
	confidence: float
	economic_risk: float
	reason: str
	votes: List[AgentVote]
	router_risk: float


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Agentic fraud detection with LLM specialists.")
	parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
	parser.add_argument("--dataset", action="append", default=[])
	parser.add_argument("--max-datasets", type=int, default=5)
	parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
	return parser.parse_args()


def load_json_file(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def safe_float(value: Any, default: float = 0.0) -> float:
	try:
		return float(value)
	except Exception:
		return default


def parse_iso_datetime(value: str) -> Optional[datetime]:
	if not value:
		return None
	text = str(value).strip()
	try:
		return datetime.fromisoformat(text)
	except Exception:
		pass

	# Fallback for message headers like "Date: 2087-03-20 13:36:49"
	for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
		try:
			return datetime.strptime(text, fmt)
		except Exception:
			continue
	return None


def normalize_text(value: str) -> str:
	if value is None:
		return ""
	decomp = unicodedata.normalize("NFKD", value)
	ascii_text = decomp.encode("ascii", "ignore").decode("ascii")
	return ascii_text.lower().strip()


def compact_spaces(text: str) -> str:
	return re.sub(r"\s+", " ", text or "").strip()


def truncate(text: str, max_len: int) -> str:
	text = compact_spaces(text)
	if len(text) <= max_len:
		return text
	return text[: max_len - 3] + "..."


def read_transactions(path: Path) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			tx = dict(row)
			tx["amount"] = safe_float(tx.get("amount"), 0.0)
			tx["timestamp_dt"] = parse_iso_datetime(tx.get("timestamp", ""))
			rows.append(tx)

	rows.sort(key=lambda x: x.get("timestamp_dt") or datetime.min)
	return rows


def parse_message_timestamp(raw_text: str) -> Optional[datetime]:
	if not raw_text:
		return None

	# Numeric format commonly used in sms.json
	m = re.search(r"Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2})", raw_text)
	if m:
		return parse_iso_datetime(m.group(1).replace(" ", "T", 1))

	# RFC 2822 style in mails.json
	m2 = re.search(r"Date:\s*(.+)", raw_text)
	if not m2:
		return None

	try:
		dt = parsedate_to_datetime(m2.group(1).strip())
		if dt is None:
			return None
		if getattr(dt, "tzinfo", None) is not None:
			return dt.replace(tzinfo=None)
		return dt
	except Exception:
		return None


def message_to_record(kind: str, raw_text: str) -> Dict[str, Any]:
	text = raw_text or ""
	return {
		"kind": kind,
		"text": text,
		"snippet": truncate(text, 220),
		"timestamp": parse_message_timestamp(text),
		"suspicious": bool(SUSPICIOUS_TEXT_RE.search(text)),
	}


def load_dataset(dataset_dir: Path) -> Dict[str, Any]:
	transactions = read_transactions(dataset_dir / "transactions.csv")
	users = load_json_file(dataset_dir / "users.json")
	locations = load_json_file(dataset_dir / "locations.json")
	mails_raw = load_json_file(dataset_dir / "mails.json")
	sms_raw = load_json_file(dataset_dir / "sms.json")

	mails = [message_to_record("mail", str(item.get("mail", ""))) for item in mails_raw]
	sms = [message_to_record("sms", str(item.get("sms", ""))) for item in sms_raw]

	audio_dir = dataset_dir / "audio"
	audio_files = []
	if audio_dir.exists() and audio_dir.is_dir():
		audio_files = [p.name for p in sorted(audio_dir.iterdir()) if p.is_file()]

	return {
		"transactions": transactions,
		"users": users,
		"locations": locations,
		"mails": mails,
		"sms": sms,
		"audio_files": audio_files,
	}


def make_user_keywords(user: Dict[str, Any]) -> List[str]:
	first = str(user.get("first_name", "")).strip()
	last = str(user.get("last_name", "")).strip()
	city = str((user.get("residence") or {}).get("city", "")).strip()
	full = f"{first} {last}".strip()
	variants = {first, last, full, city}
	return [normalize_text(v) for v in variants if v]


def assign_messages_to_users(
	users: List[Dict[str, Any]],
	all_messages: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
	by_iban: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

	user_keywords = {
		str(user.get("iban", "")): make_user_keywords(user)
		for user in users
		if user.get("iban")
	}

	for message in all_messages:
		text_norm = normalize_text(message.get("text", ""))
		best_iban = None
		best_score = 0

		for iban, keywords in user_keywords.items():
			score = 0
			for keyword in keywords:
				if keyword and keyword in text_norm:
					score += 1
			if score > best_score:
				best_score = score
				best_iban = iban

		if best_iban and best_score > 0:
			by_iban[best_iban].append(message)

	for iban in by_iban:
		by_iban[iban].sort(key=lambda x: x.get("timestamp") or datetime.min)

	return by_iban


def build_profiles(data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
	users = data["users"]
	transactions = data["transactions"]
	locations = data["locations"]

	user_by_iban: Dict[str, Dict[str, Any]] = {
		str(u.get("iban")): u for u in users if u.get("iban")
	}

	profiles: Dict[str, Dict[str, Any]] = {}
	for iban, user in user_by_iban.items():
		profiles[iban] = {
			"user": user,
			"outgoing": [],
			"incoming": [],
			"locations": [],
			"comms": [],
			"audio_files": [],
			"account_ids": set(),
			"tx_index": {},
		}

	tx_to_user_iban: Dict[str, str] = {}
	biotag_to_iban: Dict[str, str] = {}

	for tx in transactions:
		sender_iban = str(tx.get("sender_iban", ""))
		recipient_iban = str(tx.get("recipient_iban", ""))
		tx_id = str(tx.get("transaction_id", ""))

		if sender_iban in profiles:
			profiles[sender_iban]["outgoing"].append(tx)
			tx_to_user_iban[tx_id] = sender_iban
			sender_id = str(tx.get("sender_id", "")).strip()
			if sender_id:
				profiles[sender_iban]["account_ids"].add(sender_id)
				biotag_to_iban[sender_id] = sender_iban

		if recipient_iban in profiles:
			profiles[recipient_iban]["incoming"].append(tx)

	for loc in locations:
		biotag = str(loc.get("biotag", ""))
		iban = biotag_to_iban.get(biotag)
		if not iban:
			continue
		record = dict(loc)
		record["timestamp_dt"] = parse_iso_datetime(str(loc.get("timestamp", "")))
		profiles[iban]["locations"].append(record)

	msg_map = assign_messages_to_users(users, data["mails"] + data["sms"])
	for iban, messages in msg_map.items():
		if iban in profiles:
			profiles[iban]["comms"] = messages

	# Optional audio naming signal for dataset3.
	audio_files = data.get("audio_files", [])
	normalized_audio_names = [normalize_text(name) for name in audio_files]
	for iban, profile in profiles.items():
		user = profile["user"]
		full_name = normalize_text(
			f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
		)
		matched = []
		for i, name in enumerate(normalized_audio_names):
			if full_name and full_name.replace(" ", "_") in name.replace(" ", "_"):
				matched.append(audio_files[i])
		profile["audio_files"] = matched

	for iban, profile in profiles.items():
		profile["outgoing"].sort(key=lambda x: x.get("timestamp_dt") or datetime.min)
		profile["incoming"].sort(key=lambda x: x.get("timestamp_dt") or datetime.min)
		profile["locations"].sort(key=lambda x: x.get("timestamp_dt") or datetime.min)
		for idx, tx in enumerate(profile["outgoing"]):
			profile["tx_index"][str(tx.get("transaction_id", ""))] = idx

	return profiles, tx_to_user_iban


def profile_stats(profile: Dict[str, Any]) -> Dict[str, Any]:
	outgoing = profile["outgoing"]
	amounts = [safe_float(tx.get("amount"), 0.0) for tx in outgoing if tx.get("amount") is not None]

	median_amount = statistics.median(amounts) if amounts else 0.0
	mean_amount = statistics.mean(amounts) if amounts else 0.0
	std_amount = statistics.pstdev(amounts) if len(amounts) > 1 else 0.0

	tx_types = defaultdict(int)
	recipients = defaultdict(int)
	for tx in outgoing:
		tx_types[str(tx.get("transaction_type", "unknown"))] += 1
		recipient = str(tx.get("recipient_iban") or tx.get("recipient_id") or "")
		if recipient:
			recipients[recipient] += 1

	top_recipients = sorted(recipients.items(), key=lambda kv: kv[1], reverse=True)[:5]
	common_types = sorted(tx_types.items(), key=lambda kv: kv[1], reverse=True)[:5]

	loc_cities = [str(item.get("city", "")) for item in profile["locations"] if item.get("city")]
	city_count = defaultdict(int)
	for city in loc_cities:
		city_count[city] += 1
	top_cities = sorted(city_count.items(), key=lambda kv: kv[1], reverse=True)[:3]

	suspicious_messages = [m for m in profile.get("comms", []) if m.get("suspicious")]

	return {
		"median_amount": round(median_amount, 2),
		"mean_amount": round(mean_amount, 2),
		"std_amount": round(std_amount, 2),
		"n_outgoing": len(outgoing),
		"n_incoming": len(profile["incoming"]),
		"common_types": common_types,
		"top_recipients": top_recipients,
		"top_cities": top_cities,
		"n_suspicious_messages": len(suspicious_messages),
		"n_audio_events": len(profile.get("audio_files", [])),
	}


def cheap_risk_signal(tx: Dict[str, Any], stats: Dict[str, Any], known_recipients: set[str]) -> float:
	score = 0.0
	amount = safe_float(tx.get("amount"), 0.0)
	median_amount = max(stats.get("median_amount", 0.0), 1.0)
	mean_amount = stats.get("mean_amount", median_amount)
	std_amount = stats.get("std_amount", 0.0)

	z = (amount - mean_amount) / (std_amount + 1.0)
	if amount > median_amount * 2.5:
		score += 0.35
	if z > 3.5:
		score += 0.25

	tx_type = str(tx.get("transaction_type", "")).lower()
	if tx_type in {"e-commerce", "withdrawal", "in-person payment"}:
		score += 0.1

	recipient_iban = str(tx.get("recipient_iban", ""))
	if recipient_iban and recipient_iban not in known_recipients:
		score += 0.2

	ts = tx.get("timestamp_dt")
	if isinstance(ts, datetime):
		if ts.hour < 5:
			score += 0.15

	desc = str(tx.get("description", ""))
	if SUSPICIOUS_TEXT_RE.search(desc):
		score += 0.25

	return min(score, 1.0)


def json_from_llm(text: str) -> Dict[str, Any]:
	if not text:
		return {}

	cleaned = text.strip()
	cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
	cleaned = re.sub(r"```$", "", cleaned).strip()

	candidates = [cleaned]
	obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
	if obj_match:
		candidates.append(obj_match.group(0))

	for candidate in candidates:
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, dict):
				return parsed
		except Exception:
			continue
	return {}


def call_json_agent(
	session_id: str,
	llm_model: Any,
	agent_name: str,
	instruction: str,
	payload: Dict[str, Any],
) -> Dict[str, Any]:
	prompt = (
		f"You are {agent_name} in an agentic fraud detection system.\n"
		f"{instruction}\n"
		"Return ONLY strict JSON with no markdown.\n"
		"JSON schema:\n"
		"{\n"
		'  "agent": "string",\n'
		'  "risk_score": 0.0,\n'
		'  "vote": "fraud|safe|uncertain",\n'
		'  "confidence": 0.0,\n'
		'  "reasons": ["short reason"]\n'
		"}\n"
		f"Data:\n{json.dumps(payload, ensure_ascii=True)}"
	)

	try:
		raw = run_llm_call(session_id, llm_model, prompt)
	except Exception as exc:
		return {
			"agent": agent_name,
			"risk_score": 0.5,
			"vote": "uncertain",
			"confidence": 0.0,
			"reasons": [f"LLM call failed: {str(exc)}"],
		}

	parsed = json_from_llm(raw)
	if not parsed:
		return {
			"agent": agent_name,
			"risk_score": 0.5,
			"vote": "uncertain",
			"confidence": 0.1,
			"reasons": ["Invalid JSON returned by LLM"],
		}

	vote = str(parsed.get("vote", "uncertain")).strip().lower()
	if vote not in {"fraud", "safe", "uncertain"}:
		vote = "uncertain"

	reasons = parsed.get("reasons")
	if not isinstance(reasons, list):
		reasons = [str(parsed.get("reason", "No reason provided"))]

	return {
		"agent": str(parsed.get("agent", agent_name)),
		"risk_score": max(0.0, min(1.0, safe_float(parsed.get("risk_score"), 0.5))),
		"vote": vote,
		"confidence": max(0.0, min(1.0, safe_float(parsed.get("confidence"), 0.5))),
		"reasons": [truncate(str(r), 180) for r in reasons[:4]],
	}


def compact_tx(tx: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"transaction_id": tx.get("transaction_id"),
		"sender_id": tx.get("sender_id"),
		"recipient_id": tx.get("recipient_id"),
		"transaction_type": tx.get("transaction_type"),
		"amount": safe_float(tx.get("amount"), 0.0),
		"location": tx.get("location"),
		"payment_method": tx.get("payment_method"),
		"sender_iban": tx.get("sender_iban"),
		"recipient_iban": tx.get("recipient_iban"),
		"description": truncate(str(tx.get("description", "")), 120),
		"timestamp": tx.get("timestamp"),
	}


def router_candidates(
	session_id: str,
	router_model: Any,
	transactions: List[Dict[str, Any]],
	profiles: Dict[str, Dict[str, Any]],
	tx_to_user_iban: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
	candidates: Dict[str, Dict[str, Any]] = {}

	for i in range(0, len(transactions), ROUTER_CHUNK_SIZE):
		chunk = transactions[i : i + ROUTER_CHUNK_SIZE]
		router_rows = []

		for tx in chunk:
			tx_id = str(tx.get("transaction_id", ""))
			user_iban = tx_to_user_iban.get(tx_id)

			stats = {
				"median_amount": 0.0,
				"mean_amount": 0.0,
				"std_amount": 0.0,
				"n_suspicious_messages": 0,
				"top_recipients": [],
			}
			known_recipients: set[str] = set()
			if user_iban and user_iban in profiles:
				stats = profile_stats(profiles[user_iban])
				known_recipients = {item[0] for item in stats.get("top_recipients", [])}

			router_rows.append(
				{
					"tx": compact_tx(tx),
					"cheap_signal": round(cheap_risk_signal(tx, stats, known_recipients), 3),
					"user_stats": {
						"median_amount": stats.get("median_amount", 0.0),
						"std_amount": stats.get("std_amount", 0.0),
						"n_suspicious_messages": stats.get("n_suspicious_messages", 0),
					},
				}
			)

		per_chunk_top = max(3, min(15, int(math.ceil(len(chunk) * 0.22))))
		instruction = (
			"Select the most suspicious transactions in this chunk for deeper analysis. "
			"Prioritize high economic risk and high fraud likelihood. "
			"Return strict JSON: {\"candidates\":[{\"transaction_id\":\"...\",\"risk\":0.0,\"why\":\"...\"}]}. "
			f"Return up to {per_chunk_top} candidates."
		)
		payload = {"rows": router_rows}

		prompt = (
			"You are router_agent in a fraud detection system.\n"
			f"{instruction}\n"
			"Return ONLY strict JSON with no markdown.\n"
			f"Data:\n{json.dumps(payload, ensure_ascii=True)}"
		)

		try:
			raw = run_llm_call(session_id, router_model, prompt)
			parsed = json_from_llm(raw)
		except Exception:
			parsed = {}

		chunk_candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []
		if not isinstance(chunk_candidates, list):
			chunk_candidates = []

		# Deterministic fallback if router output is malformed.
		if not chunk_candidates:
			sorted_rows = sorted(router_rows, key=lambda r: r["cheap_signal"], reverse=True)
			for row in sorted_rows[:per_chunk_top]:
				tx_id = str((row.get("tx") or {}).get("transaction_id", ""))
				if not tx_id:
					continue
				chunk_candidates.append(
					{
						"transaction_id": tx_id,
						"risk": float(row.get("cheap_signal", 0.5)),
						"why": "Fallback risk shortlist",
					}
				)

		for item in chunk_candidates:
			tx_id = str(item.get("transaction_id", ""))
			if not tx_id:
				continue
			risk = max(0.0, min(1.0, safe_float(item.get("risk"), 0.5)))
			why = truncate(str(item.get("why", "")), 160)

			prev = candidates.get(tx_id)
			if prev is None or risk > prev["risk"]:
				candidates[tx_id] = {"risk": risk, "why": why}

	return candidates


def tx_neighbors(profile: Dict[str, Any], tx_id: str) -> List[Dict[str, Any]]:
	idx = profile.get("tx_index", {}).get(tx_id)
	if idx is None:
		return []
	outgoing = profile.get("outgoing", [])
	start = max(0, idx - MAX_RECENT_TX_CONTEXT)
	end = min(len(outgoing), idx + 1)
	return outgoing[start:end]


def geo_context(profile: Dict[str, Any], tx: Dict[str, Any]) -> Dict[str, Any]:
	home_city = str((profile.get("user", {}).get("residence") or {}).get("city", ""))
	tx_ts = tx.get("timestamp_dt")

	nearest = []
	if isinstance(tx_ts, datetime):
		for loc in profile.get("locations", []):
			ts = loc.get("timestamp_dt")
			if not isinstance(ts, datetime):
				continue
			if abs((tx_ts - ts).total_seconds()) <= 24 * 3600:
				nearest.append(
					{
						"timestamp": str(loc.get("timestamp", "")),
						"city": str(loc.get("city", "")),
						"lat": loc.get("lat"),
						"lng": loc.get("lng"),
					}
				)
	nearest = nearest[:8]

	return {
		"home_city": home_city,
		"tx_location": str(tx.get("location", "")),
		"recent_location_observations": nearest,
	}


def comms_context(profile: Dict[str, Any], tx: Dict[str, Any]) -> Dict[str, Any]:
	tx_ts = tx.get("timestamp_dt")
	comms = profile.get("comms", [])

	recent = []
	suspicious_count_recent = 0
	suspicious_count_30d = 0
	for item in comms:
		ts = item.get("timestamp")
		if not isinstance(ts, datetime):
			continue
		if not isinstance(tx_ts, datetime):
			continue

		delta_seconds = (tx_ts - ts).total_seconds()
		if delta_seconds < 0:
			continue

		delta_days = delta_seconds / 86400.0
		include = delta_days <= 30
		if item.get("suspicious") and include:
			suspicious_count_30d += 1

		if include:
			recent.append(
				{
					"kind": item.get("kind"),
					"timestamp": ts.isoformat() if isinstance(ts, datetime) else None,
					"suspicious": bool(item.get("suspicious")),
					"snippet": item.get("snippet"),
				}
			)
			if item.get("suspicious") and delta_days <= 7:
				suspicious_count_recent += 1

		if len(recent) >= MAX_COMMS_CONTEXT:
			break

	total_suspicious = sum(1 for item in comms if item.get("suspicious"))

	return {
		"recent_messages": recent,
		"recent_suspicious_message_count": suspicious_count_recent,
		"suspicious_message_count_30d": suspicious_count_30d,
		"total_messages": len(comms),
		"total_suspicious_messages": total_suspicious,
	}


def description_key(description: str) -> str:
	clean = normalize_text(description or "")
	clean = re.sub(
		r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b",
		"",
		clean,
	)
	clean = re.sub(r"[^a-z0-9 ]+", " ", clean)
	return compact_spaces(clean)


def location_city_token(location: str) -> str:
	text = str(location or "").strip()
	if not text:
		return ""
	parts = text.split(" - ")
	if parts:
		return normalize_text(parts[0])
	return normalize_text(text)


def candidate_features(profile: Dict[str, Any], tx: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
	tx_id = str(tx.get("transaction_id", ""))
	outgoing = profile.get("outgoing", [])
	idx = profile.get("tx_index", {}).get(tx_id)
	if idx is None:
		idx = len(outgoing)
	prior = outgoing[:idx]

	amount = safe_float(tx.get("amount"), 0.0)
	median_amount = max(stats.get("median_amount", 0.0), 1.0)
	mean_amount = stats.get("mean_amount", median_amount)
	std_amount = stats.get("std_amount", 0.0)
	amount_ratio = amount / median_amount
	z_score = (amount - mean_amount) / (std_amount + 1.0)

	recipient_iban = str(tx.get("recipient_iban", ""))
	tx_type = str(tx.get("transaction_type", "")).lower()
	description = str(tx.get("description", ""))
	desc_key = description_key(description)

	same_recipient_prior = [
		p
		for p in prior
		if str(p.get("recipient_iban", "")) == recipient_iban
		and str(p.get("transaction_type", "")).lower() == tx_type
	]

	similar_amount_prior = [
		p
		for p in same_recipient_prior
		if abs(safe_float(p.get("amount"), 0.0) - amount) <= max(20.0, amount * 0.12)
	]

	same_desc_prior = [
		p
		for p in same_recipient_prior
		if description_key(str(p.get("description", ""))) == desc_key and desc_key
	]

	recurring_pattern = (
		len(same_recipient_prior) >= 2 and (len(similar_amount_prior) >= 2 or len(same_desc_prior) >= 1)
	)

	tx_ts = tx.get("timestamp_dt")
	off_hours = isinstance(tx_ts, datetime) and tx_ts.hour < 5

	burst_count = 0
	if isinstance(tx_ts, datetime):
		for p in prior[-12:]:
			pts = p.get("timestamp_dt")
			if isinstance(pts, datetime) and 0 <= (tx_ts - pts).total_seconds() <= 90 * 60:
				burst_count += 1

	cc = comms_context(profile, tx)
	suspicious_7d = int(cc.get("recent_suspicious_message_count", 0))
	suspicious_30d = int(cc.get("suspicious_message_count_30d", 0))

	home_city = normalize_text(str((profile.get("user", {}).get("residence") or {}).get("city", "")))
	tx_city = location_city_token(str(tx.get("location", "")))
	geo_nearby_match = False
	if isinstance(tx_ts, datetime) and tx_city:
		for loc in profile.get("locations", []):
			lts = loc.get("timestamp_dt")
			if not isinstance(lts, datetime):
				continue
			if abs((tx_ts - lts).total_seconds()) <= 8 * 3600:
				if normalize_text(str(loc.get("city", ""))) == tx_city:
					geo_nearby_match = True
					break

	geo_mismatch = bool(tx_city) and tx_city != home_city and not geo_nearby_match

	pm = normalize_text(str(tx.get("payment_method", "")))
	new_recipient = len(same_recipient_prior) == 0 and bool(recipient_iban)

	return {
		"amount": amount,
		"amount_ratio": round(amount_ratio, 3),
		"z_score": round(z_score, 3),
		"new_recipient": new_recipient,
		"same_recipient_prior_count": len(same_recipient_prior),
		"recurring_pattern": recurring_pattern,
		"suspicious_7d": suspicious_7d,
		"suspicious_30d": suspicious_30d,
		"off_hours": bool(off_hours),
		"burst_count": burst_count,
		"geo_mismatch": geo_mismatch,
		"payment_method": pm,
		"tx_type": tx_type,
		"description_key": desc_key,
		"recent_messages": cc.get("recent_messages", []),
	}


def deterministic_risk_score(features: Dict[str, Any]) -> float:
	risk = 0.0
	amount_ratio = float(features.get("amount_ratio", 0.0))
	z_score = float(features.get("z_score", 0.0))
	tx_type = str(features.get("tx_type", ""))
	desc_key = str(features.get("description_key", ""))
	pm = str(features.get("payment_method", ""))

	if features.get("new_recipient") and amount_ratio >= 1.7:
		risk += 0.23
	if amount_ratio >= 2.4:
		risk += 0.16
	if z_score >= 2.8:
		risk += 0.12
	if tx_type in {"e-commerce", "withdrawal", "in-person payment"} and amount_ratio >= 1.2:
		risk += 0.1
	if int(features.get("suspicious_7d", 0)) > 0:
		risk += 0.2
	if int(features.get("suspicious_30d", 0)) > 0:
		risk += min(0.12, 0.03 * int(features.get("suspicious_30d", 0)))
	if features.get("off_hours"):
		risk += 0.08
	if int(features.get("burst_count", 0)) >= 3:
		risk += 0.08
	if features.get("geo_mismatch"):
		risk += 0.1
	if tx_type == "e-commerce" and features.get("new_recipient") and pm in {"paypal", "google pay"}:
		risk += 0.08

	# Strong false-positive suppressors for recurring salary/rent/utilities style traffic.
	if features.get("recurring_pattern"):
		risk -= 0.34
	if "rent payment" in desc_key and features.get("recurring_pattern"):
		risk -= 0.2
	if "salary payment" in desc_key:
		risk -= 0.45
	if tx_type == "direct debit" and features.get("recurring_pattern"):
		risk -= 0.18
	if int(features.get("same_recipient_prior_count", 0)) >= 5 and amount_ratio <= 1.5:
		risk -= 0.15

	return max(0.0, min(1.0, risk))


def vote_from_payload(payload: Dict[str, Any]) -> AgentVote:
	return AgentVote(
		agent=str(payload.get("agent", "unknown")),
		risk_score=max(0.0, min(1.0, safe_float(payload.get("risk_score"), 0.5))),
		vote=str(payload.get("vote", "uncertain")),
		confidence=max(0.0, min(1.0, safe_float(payload.get("confidence"), 0.5))),
		reasons=[str(x) for x in payload.get("reasons", [])][:4],
	)


def judge_decision(
	session_id: str,
	judge_model: Any,
	tx: Dict[str, Any],
	user_summary: Dict[str, Any],
	features: Dict[str, Any],
	recent_transactions: List[Dict[str, Any]],
	recent_messages: List[Dict[str, Any]],
	router_risk: float,
	stage: str,
) -> Decision:
	payload = {
		"transaction": compact_tx(tx),
		"user_summary": user_summary,
		"features": features,
		"recent_transactions": recent_transactions,
		"recent_messages": recent_messages,
		"router_risk": router_risk,
		"stage": stage,
	}

	prompt = (
		"You are judge_agent in an agentic fraud system.\n"
		"Decide if this user-owned outgoing transaction is fraud.\n"
		"Prioritize precision and economic impact.\n"
		"Recurring rent/salary/utilities-like patterns are usually NOT fraud.\n"
		"Output ONLY strict JSON with this schema:\n"
		"{\n"
		'  "is_fraud": true,\n'
		'  "risk_score": 0.0,\n'
		'  "confidence": 0.0,\n'
		'  "economic_risk": 0.0,\n'
		'  "reason": "very short explanation"\n'
		"}\n"
		"`economic_risk` must be in [0, amount].\n"
		f"Data:\n{json.dumps(payload, ensure_ascii=True)}"
	)

	try:
		raw = run_llm_call(session_id, judge_model, prompt)
		parsed = json_from_llm(raw)
	except Exception:
		parsed = {}

	amount = safe_float(tx.get("amount"), 0.0)
	if not parsed:
		is_fraud = router_risk >= 0.7
		conf = max(0.25, min(0.8, router_risk))
		econ = amount * (0.55 + 0.35 * conf) if is_fraud else amount * 0.03
		reason = "Fallback decision from deterministic risk"
		vote = AgentVote(
			agent="judge_agent_fallback",
			risk_score=router_risk,
			vote="fraud" if is_fraud else "safe",
			confidence=conf,
			reasons=[reason],
		)
		return Decision(
			transaction_id=str(tx.get("transaction_id", "")),
			amount=amount,
			is_fraud=is_fraud,
			confidence=conf,
			economic_risk=min(amount, max(0.0, econ)),
			reason=reason,
			votes=[vote],
			router_risk=router_risk,
		)

	is_fraud = bool(parsed.get("is_fraud", False))
	risk_score = max(0.0, min(1.0, safe_float(parsed.get("risk_score"), router_risk)))
	confidence = max(0.0, min(1.0, safe_float(parsed.get("confidence"), 0.5)))
	economic_risk = max(0.0, min(amount, safe_float(parsed.get("economic_risk"), amount * confidence)))
	reason = truncate(str(parsed.get("reason", "No reason provided")), 220)
	vote = AgentVote(
		agent="judge_agent",
		risk_score=risk_score,
		vote="fraud" if is_fraud else "safe",
		confidence=confidence,
		reasons=[reason],
	)

	return Decision(
		transaction_id=str(tx.get("transaction_id", "")),
		amount=amount,
		is_fraud=is_fraud,
		confidence=confidence,
		economic_risk=economic_risk,
		reason=reason,
		votes=[vote],
		router_risk=router_risk,
	)


def evaluate_candidate(
	session_id: str,
	tx: Dict[str, Any],
	router_risk: float,
	profile: Dict[str, Any],
	judge_model: Any,
	reviewer_model: Any,
	stats: Dict[str, Any],
	features: Dict[str, Any],
) -> Decision:
	neighbors = tx_neighbors(profile, str(tx.get("transaction_id", "")))
	prior_neighbors = [compact_tx(x) for x in neighbors[:-1]][-6:]
	user = profile.get("user", {})

	user_summary = {
		"first_name": user.get("first_name"),
		"last_name": user.get("last_name"),
		"city": (user.get("residence") or {}).get("city"),
		"salary": user.get("salary"),
		"job": user.get("job"),
		"stats": {
			"median_amount": stats.get("median_amount", 0.0),
			"mean_amount": stats.get("mean_amount", 0.0),
			"std_amount": stats.get("std_amount", 0.0),
			"n_outgoing": stats.get("n_outgoing", 0),
			"top_recipients": stats.get("top_recipients", [])[:3],
		},
		"audio_event_count": len(profile.get("audio_files", [])),
	}

	comms_ctx = comms_context(profile, tx)
	decision = judge_decision(
		session_id=session_id,
		judge_model=judge_model,
		tx=tx,
		user_summary=user_summary,
		features={
			k: v
			for k, v in features.items()
			if k not in {"recent_messages"}
		},
		recent_transactions=prior_neighbors,
		recent_messages=comms_ctx.get("recent_messages", []),
		router_risk=router_risk,
		stage="primary",
	)

	# Optional second-pass reviewer for uncertain, economically relevant cases.
	amount = safe_float(tx.get("amount"), 0.0)
	median_amount = max(stats.get("median_amount", 0.0), 1.0)
	requires_review = (
		0.42 <= decision.confidence <= 0.68
		and router_risk >= 0.52
		and amount >= (1.6 * median_amount)
	)

	if requires_review and reviewer_model is not None:
		review = judge_decision(
			session_id=session_id,
			judge_model=reviewer_model,
			tx=tx,
			user_summary=user_summary,
			features={
				"primary_decision": {
					"is_fraud": decision.is_fraud,
					"confidence": decision.confidence,
					"reason": decision.reason,
				},
				**{k: v for k, v in features.items() if k not in {"recent_messages"}},
			},
			recent_transactions=prior_neighbors,
			recent_messages=comms_ctx.get("recent_messages", []),
			router_risk=router_risk,
			stage="review",
		)

		# Blend confidence conservatively to avoid over-triggering.
		decision.is_fraud = bool(review.is_fraud and decision.is_fraud)
		decision.confidence = (0.55 * decision.confidence) + (0.45 * review.confidence)
		decision.reason = truncate(f"{decision.reason}; review: {review.reason}", 220)
		decision.economic_risk = max(decision.economic_risk, review.economic_risk)
		decision.votes.extend(review.votes)

	return decision


def process_dataset(
	dataset_dir: Path,
	output_dir: Path,
	session_id: str,
	max_workers: int,
) -> Dict[str, Any]:
	data = load_dataset(dataset_dir)
	profiles, tx_to_user_iban = build_profiles(data)
	transactions = data["transactions"]
	candidate_pool = [
		tx
		for tx in transactions
		if str(tx.get("transaction_id", "")) in tx_to_user_iban
	]

	judge_model = get_model()
	reviewer_model = get_model()

	if not judge_model:
		raise RuntimeError("LLM model is not configured. Set OPENROUTER_API_KEY and MODEL_ID.")

	# Deterministic shortlisting keeps cost low; LLM remains core in final judging.
	candidate_info: Dict[str, Dict[str, Any]] = {}
	for tx in candidate_pool:
		tx_id = str(tx.get("transaction_id", ""))
		user_iban = tx_to_user_iban.get(tx_id)
		profile = profiles.get(user_iban or "")
		if not profile:
			continue

		stats = profile_stats(profile)
		features = candidate_features(profile, tx, stats)
		risk = deterministic_risk_score(features)
		candidate_info[tx_id] = {
			"risk": risk,
			"stats": stats,
			"features": features,
		}

	scored = sorted(candidate_info.items(), key=lambda kv: kv[1]["risk"], reverse=True)
	desired = int(max(MIN_CANDIDATES, min(MAX_CANDIDATES, len(candidate_pool) * CANDIDATE_RATIO)))
	desired = min(desired, len(scored))
	shortlisted_ids = {tx_id for tx_id, _ in scored[:desired]}

	# Always include very high deterministic-risk candidates.
	for tx_id, info in scored:
		if float(info.get("risk", 0.0)) >= 0.76:
			shortlisted_ids.add(tx_id)

	shortlisted = {tx_id: candidate_info[tx_id] for tx_id in shortlisted_ids if tx_id in candidate_info}

	tx_by_id = {str(tx.get("transaction_id", "")): tx for tx in transactions}

	decisions: List[Decision] = []
	futures = []
	with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
		for tx_id, metadata in shortlisted.items():
			tx = tx_by_id.get(tx_id)
			user_iban = tx_to_user_iban.get(tx_id)
			profile = profiles.get(user_iban or "")
			if not tx or not profile:
				continue
			futures.append(
				executor.submit(
					evaluate_candidate,
					session_id,
					tx,
					float(metadata.get("risk", 0.5)),
					profile,
					judge_model,
					reviewer_model,
					metadata.get("stats", {}),
					metadata.get("features", {}),
				)
			)

		for future in as_completed(futures):
			decisions.append(future.result())

	decisions.sort(key=lambda d: d.router_risk, reverse=True)
	fraud_decisions = []
	for d in decisions:
		if not d.is_fraud:
			continue
		combined = (0.65 * d.confidence) + (0.35 * d.router_risk)
		conservative_hit = combined >= 0.58 and d.router_risk >= 0.24
		high_impact_hit = (
			d.confidence >= 0.72
			and d.router_risk >= 0.18
			and d.economic_risk >= d.amount * 0.6
		)
		if conservative_hit or high_impact_hit:
			fraud_decisions.append(d)
	fraud_ids = sorted({d.transaction_id for d in fraud_decisions})

	output_dir.mkdir(parents=True, exist_ok=True)
	ids_file = output_dir / f"{dataset_dir.name}_fraud_ids.txt"
	ids_file.write_text("\n".join(fraud_ids) + ("\n" if fraud_ids else ""), encoding="ascii", errors="ignore")

	summary = get_session_summary(session_id) or {}
	meta = {
		"dataset": dataset_dir.name,
		"session_id": session_id,
		"total_transactions": len(transactions),
		"candidate_transactions": len(shortlisted),
		"predicted_fraud_count": len(fraud_ids),
		"predicted_fraud_economic_sum": round(sum(d.economic_risk for d in fraud_decisions), 2),
		"top_decisions": [
			{
				"transaction_id": d.transaction_id,
				"is_fraud": d.is_fraud,
				"confidence": d.confidence,
				"economic_risk": d.economic_risk,
				"router_risk": d.router_risk,
				"reason": d.reason,
			}
			for d in decisions[:30]
		],
		"metrics": summary,
	}
	meta_file = output_dir / f"{dataset_dir.name}_fraud_meta.json"
	meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

	return {
		"dataset": dataset_dir.name,
		"ids_file": str(ids_file),
		"meta_file": str(meta_file),
		"total_transactions": len(transactions),
		"candidate_count": len(shortlisted),
		"predicted_fraud_count": len(fraud_ids),
		"predicted_economic_risk": round(sum(d.economic_risk for d in fraud_decisions), 2),
	}


def resolve_datasets(data_root: Path, explicit: Iterable[str], max_datasets: int) -> List[Path]:
	if explicit:
		paths = [data_root / name for name in explicit]
	else:
		paths = sorted([p for p in data_root.iterdir() if p.is_dir()])
	return [p for p in paths if p.exists() and p.is_dir()][:max(1, max_datasets)]


def main() -> None:
	args = parse_args()
	data_root = Path(args.data_root)
	output_root = Path(args.output_dir)

	datasets = resolve_datasets(data_root, args.dataset, args.max_datasets)
	if not datasets:
		raise FileNotFoundError(f"No dataset folders found under: {data_root}")

	run_summaries = []
	total_cost = 0.0
	total_latency = 0.0

	for dataset_dir in datasets:
		session_id = generate_session_id()
		print(f"\n=== Processing {dataset_dir.name} | session_id={session_id} ===")

		result = process_dataset(
			dataset_dir=dataset_dir,
			output_dir=output_root,
			session_id=session_id,
			max_workers=args.max_workers,
		)
		run_summaries.append(result)

		print(
			f"Dataset={result['dataset']} "
			f"transactions={result['total_transactions']} "
			f"candidates={result['candidate_count']} "
			f"fraud_predictions={result['predicted_fraud_count']} "
			f"economic_risk={result['predicted_economic_risk']}"
		)
		print(f"Fraud IDs file: {result['ids_file']}")
		print(f"Meta file: {result['meta_file']}")

		trace_info = get_trace_info(session_id)
		if trace_info:
			print_results(trace_info, session_id=session_id)
			total_cost += float(getattr(trace_info, "total_cost", 0.0) or 0.0)
			total_latency += float(getattr(trace_info, "latency", 0.0) or 0.0)
		else:
			local = get_session_summary(session_id) or {}
			print_results(local, session_id=session_id)
			total_cost += float(local.get("total_cost", 0.0))
			total_latency += float(local.get("latency", 0.0))

	print("\n=== Global Run Summary ===")
	print(f"Datasets processed: {len(run_summaries)}")
	print(f"Aggregate latency: {total_latency:.3f}s")
	print(f"Aggregate total cost: ${total_cost:.6f}")


if __name__ == "__main__":
	main()
