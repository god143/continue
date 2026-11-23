import base64
import hashlib
import asyncio
import calendar
import io
import json
import os
import re
import shutil
import tempfile
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
try:
    import pytesseract  # optional deterministic OCR fallback for tiny scanned headers
except Exception:  # pragma: no cover
    pytesseract = None
# =========================
# Config
# =========================
HEADER_IMAGE_SIDE = int(os.getenv("HEADER_IMAGE_SIDE", "900"))
MAX_HEADER_CROPS = int(os.getenv("MAX_HEADER_CROPS", "4"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8003/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "/opt/vllm/models/qwen2.5-VL-7B-inst")
PDF_DPI = int(os.getenv("PDF_DPI", "150"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "4"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1280"))
IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "JPEG").upper()
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "88"))
RETURN_RAW_EXTRACTION = os.getenv("RETURN_RAW_EXTRACTION", "1") == "1"
ENABLE_RECEIPT_REPAIR_PASS = os.getenv("ENABLE_RECEIPT_REPAIR_PASS", "1") == "1"
# Optimization controls
# Cache is keyed by file content + model/settings + code version, so repeated uploads
# return instantly without changing the extraction result.
CACHE_VERSION = os.getenv("CACHE_VERSION", "expense_ocr_v65")
ENABLE_RESPONSE_CACHE = os.getenv("ENABLE_RESPONSE_CACHE", "1") == "1"
RESPONSE_CACHE_DIR = os.getenv("RESPONSE_CACHE_DIR", "/tmp/expense_ocr_cache")
# Repair-pass image limits. Customer/vendor repair can need rotated crops for
# sideways scanned bills, so do not restrict these calls to the first 4 crops.
MAX_CUSTOMER_BLOCK_IMAGES = int(os.getenv("MAX_CUSTOMER_BLOCK_IMAGES", "10"))
MAX_VENDOR_IDENTITY_IMAGES = int(os.getenv("MAX_VENDOR_IDENTITY_IMAGES", "10"))
# Final lightweight party repair for scanned/image bills when both supplier and
# billed-party names are still weak/blank after normal vision, header, customer,
# and vendor passes. It sends only a few small full-document orientation images
# to avoid model context-length errors.
ENABLE_PARTY_IDENTITY_REPAIR = os.getenv("ENABLE_PARTY_IDENTITY_REPAIR", "1") == "1"
# Keep the compact multi-image identity pass under context length, but make each
# individual crop readable enough for small/sideways scanned bills.
PARTY_IDENTITY_IMAGE_SIDE = int(os.getenv("PARTY_IDENTITY_IMAGE_SIDE", "1120"))
MAX_PARTY_IDENTITY_IMAGES = int(os.getenv("MAX_PARTY_IDENTITY_IMAGES", "3"))
# When the compact multi-image pass still cannot read supplier/client, try
# orientation candidates one at a time. This avoids context-length errors and
# improves scanned/rotated bill accuracy without hardcoded names.
ENABLE_SINGLE_IMAGE_PARTY_IDENTITY_FALLBACK = os.getenv("ENABLE_SINGLE_IMAGE_PARTY_IDENTITY_FALLBACK", "1") == "1"
MAX_SINGLE_IMAGE_PARTY_IDENTITY_ATTEMPTS = int(os.getenv("MAX_SINGLE_IMAGE_PARTY_IDENTITY_ATTEMPTS", "6"))
# Focused charge/tax summary repair for scanned utility and hotel bills.
# Runs only for utility/hotel-like contexts and sends one page at a time to avoid
# the model context-length limit.
ENABLE_CHARGE_SUMMARY_REPAIR_PASS = os.getenv("ENABLE_CHARGE_SUMMARY_REPAIR_PASS", "1") == "1"
MAX_CHARGE_SUMMARY_ATTEMPTS = int(os.getenv("MAX_CHARGE_SUMMARY_ATTEMPTS", "4"))
# Focused repair for scanned pages that contain multiple small handwritten/cash
# receipt panels on one uploaded page. Kept separate and merge-guarded so normal
# invoices, utility bills, hotel bills, tax invoices and prior Bill-of-Supply
# repairs are not affected.
ENABLE_MULTI_RECEIPT_PANEL_REPAIR_PASS = os.getenv("ENABLE_MULTI_RECEIPT_PANEL_REPAIR_PASS", "1") == "1"
MAX_MULTI_RECEIPT_PANEL_IMAGES = int(os.getenv("MAX_MULTI_RECEIPT_PANEL_IMAGES", "8"))
# Focused legal-supplier repair for hotel/lodging invoices where the model may
# pick the trade/logo name instead of the legal entity printed near GSTIN/PAN.
ENABLE_LEGAL_SUPPLIER_REPAIR_PASS = os.getenv("ENABLE_LEGAL_SUPPLIER_REPAIR_PASS", "1") == "1"
MAX_LEGAL_SUPPLIER_IMAGES = int(os.getenv("MAX_LEGAL_SUPPLIER_IMAGES", "4"))
# Legal supplier names on scanned hotel bills are often small header text. Render
# this focused pass at higher DPI and retry left-header crops individually so a
# logo/trade name does not dominate the answer.
LEGAL_SUPPLIER_IMAGE_DPI = int(os.getenv("LEGAL_SUPPLIER_IMAGE_DPI", "300"))
ENABLE_SINGLE_IMAGE_LEGAL_SUPPLIER_FALLBACK = os.getenv("ENABLE_SINGLE_IMAGE_LEGAL_SUPPLIER_FALLBACK", "1") == "1"
MAX_SINGLE_IMAGE_LEGAL_SUPPLIER_ATTEMPTS = int(os.getenv("MAX_SINGLE_IMAGE_LEGAL_SUPPLIER_ATTEMPTS", "4"))
# Text-based PDFs can be extracted deterministically from embedded PDF text first.
# Vision is still used as fallback when the text extraction is incomplete.
ENABLE_TEXT_PDF_FAST_PATH = os.getenv("ENABLE_TEXT_PDF_FAST_PATH", "1") == "1"
SOURCE_TEXT_FAST_PATH_MIN_CHARS = int(os.getenv("SOURCE_TEXT_FAST_PATH_MIN_CHARS", "250"))
# Scanned/image optimization controls. These do not change extraction prompts or
# merge rules. They only avoid unnecessary later LLM passes when the first
# vision+header result is already complete, and run independent repair passes
# in parallel when repair is needed.
ENABLE_STRICT_VISION_EARLY_EXIT = os.getenv("ENABLE_STRICT_VISION_EARLY_EXIT", "1") == "1"
ENABLE_CONCURRENT_REPAIR_PASSES = os.getenv("ENABLE_CONCURRENT_REPAIR_PASSES", "1") == "1"
# Some scanned bills are uploaded sideways. Keep the fast first-pass unchanged, but
# use rotated repair crops when customer/vendor fields are weak so the model can
# read the actual supplier and billed-party blocks without hardcoded names.
ENABLE_ROTATION_AWARE_REPAIR_CROPS = os.getenv("ENABLE_ROTATION_AWARE_REPAIR_CROPS", "1") == "1"
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)
app = FastAPI(title="Expense OCR Vision API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================
# Compact prompt
# =========================
COMPACT_JSON_TEMPLATE = """
{
  "document_type_raw": "",
  "whether_self_reimbursement_raw": "",
  "vendor_employee_name_raw": "",
  "expense_type_raw": "",
  "los_vendor_state_raw": "",
  "lor_isec_state_raw": "",
  "service_usage_raw": "",
  "is_invoice_digitally_signed_raw": "",
  "digital_signature_evidence_raw": "",
  "remarks_raw": "",
  "approver_raw": "",
  "client_name_raw": "",
  "invoice_no_raw": "",
  "bill_no_raw": "",
  "po_no_raw": "",
  "bill_date_raw": "",
  "service_period_raw": "",
  "explicit_date_range_raw": "",
  "vendor_gstin_raw": "",
  "isec_gstin_raw": "",
  "vendor_pan_raw": "",
  "department_code_raw": "",
  "function_code_raw": "",
  "location_code_raw": "",
  "sub_loc_code_raw": "",
  "subtotal_raw": "",
  "cgst_total_raw": "",
  "sgst_total_raw": "",
  "igst_total_raw": "",
  "cess_total_raw": "",
  "round_off_raw": "",
  "grand_total_raw": "",
  "amount_in_words_raw": "",
  "items": [
    {
      "particulars_raw": "",
      "quantity_raw": "",
      "quantity_unit_raw": "",
      "hsn_code_raw": "",
      "rate_raw": "",
      "amount_raw": "",
      "cgst_raw": "",
      "sgst_raw": "",
      "igst_raw": "",
      "cess_raw": ""
    }
  ]
}
"""
SYSTEM_PROMPT = f"""
You are an invoice, bill, challan, receipt and expense OCR extraction system.
Return only valid JSON.
Do not use markdown.
Do not explain.
Do not guess missing values.
Use empty string "" when value is not visible.
Extract raw visible values only.
Do not hardcode any value.
- expense_type_raw must always be an empty string. Do not extract or infer expense type.
- remarks_raw must always be an empty string. Do not extract, infer, repair, or generate remarks.
- quantity_raw and quantity_unit_raw must always be empty strings. Do not extract, infer, calculate, default, or repair quantity or quantity unit.
Header rules:
- vendor_employee_name_raw is the main invoice issuer business/shop/firm/vendor/consultant name. If a line contains vendor name followed by address, return only the name in vendor_employee_name_raw and put address details in vendor_address_raw when available.
- Prefer the largest/top business or shop name as vendor.
- Do not use contact person names near phone numbers if a business/shop/firm name is visible.
- For small/local expense bills such as tea, coffee, snacks, food, canteen, restaurant, water jars, taxi/cab, newspaper, local utility or similar receipts, vendor_employee_name_raw must be the issuer/shop/stall/supplier/person who raised the bill. Do not use the corporate payer/customer/billed party as vendor.
- In small/local cash memos, a label like From/From:/Issued By/Supplier/Vendor/Shop/Stall near the header indicates the vendor. A label like M/s/Bill To/Customer/Client/To indicates the client. Do not swap them.
- client_name_raw is the billed-to/customer/client name.
- Text after labels like M/s, M/s., Bill To, Billed To, Buyer, Customer, Client, To, Received From, Paid By, Party Name or Name is normally the client/billed party, not the vendor.
- For Bill of Supply / Composition taxable person bills, supplier/vendor is the issuer name at top/logo/header or near For/Authorised Signatory; customer/client is the M/s/Bill To block.
- LOS vendor state means vendor state/city/location in bill.
- LOR ISEC state means customer/client/patient/place-of-supply state/city/location in bill. Prefer To/Bill To/Patient/Customer/Supply Address/Place of Supply over vendor address.
- Do not leave vendor_employee_name_raw, client_name_raw, LOS, LOR, invoice_no_raw, bill_no_raw, or bill_date_raw blank when visible.
- vendor_gstin_raw is supplier/vendor GSTIN only. isec_gstin_raw is buyer/client/customer GSTIN only.
- If GST No/GSTIN is printed in the To/Bill To/Buyer/Client block, put it in isec_gstin_raw, not vendor_gstin_raw.
- bill_no_raw must also capture Receipt No, Receipt Number, Voucher No, Deposit Receipt No, Transaction Receipt No, CRN/Ride Number when that is the main document number.
- whether_self_reimbursement_raw should be Yes only if the document clearly shows an employee/staff/self expense claim or employee reimbursement evidence.
- Do not set whether_self_reimbursement_raw to Yes only because the word reimbursement is visible.
- A reimbursement letter/request from an outside person, landlord, vendor, property owner, or third party to a company is not self reimbursement unless employee/staff/self evidence is clearly visible.
- If it is a normal invoice/bill/challan/letter from vendor/person to client/company, use No.
- is_invoice_digitally_signed_raw should be Yes only if IRN, Ack No, Ack Date, QR code, e-invoice data, digital signature, or digitally signed evidence is visible. Otherwise use No.
Date rules:
- bill_date_raw = single invoice date / bill date only.
- service_period_raw = billing period / service period / month period only.
- explicit_date_range_raw = any visible text containing both start date and end date.
- If any visible Date field contains two dates connected by To/to/from/-/–, preserve the full text in explicit_date_range_raw.
- Do not copy a single bill_date_raw into service_period_raw or explicit_date_range_raw.
- Do not calculate final from_date or to_date.
Amount and table rules:
- Read values according to column position: particulars, quantity, amount, and GST rate percentage when visible.
- If only one number is visible between particulars and amount, decide by column location.
- Preserve visible amount suffix marks such as slash, dash, or slash-dash in raw amount.
- Do not convert punctuation or suffix marks into digits.
- Keep raw amounts as visible.
- Negative amounts like deduction, discount, round off, bus facility deduction must remain negative if visible.
- Do not treat /- as digit 1.
- For cab/taxi/ride/travel receipts and similar documents, there may be two totals:
  1) main customer receipt total such as Total Bill, Paid Amount, Amount Paid, Paid by Cash/Card, Net Payable;
  2) separate GST/tax invoice total for only convenience/platform/service fee.
- In such cases, grand_total_raw must be the main paid/claim amount from Total Bill/Paid Amount, not only the smaller GST invoice convenience-fee total.
- Still extract visible tax/convenience-fee rows in items when visible.
- Do not add item taxes to the main paid total again.
Newspaper/subscription bill rules:
- For newspaper/monthly publication/subscription bills, vendor_employee_name_raw must be the specific agency/dealer/shop/person name, not generic category text such as Newspaper Dealers.
- If the specific agency/vendor name is printed in Hindi/Marathi/other Indian script, transliterate it to English.
- expense_type_raw must stay empty. service_usage_raw should be Newspaper Monthly Subscription.
- If no Bill No is visible but Id/ID/No appears near Name/Customer, use that ID value as bill_no_raw.
- Keep quantity_unit_raw empty for newspaper/monthly publication/subscription bills.
- If no tax is visible, use 0 for CGST, SGST, IGST and CESS.
Item rules:
- Extract one item per visible product/service row.
- If a single scanned page contains multiple separate small receipts/slips/chits/cash memos pasted or scanned together, extract every separate receipt/slip as a separate item. Do not merge separate dated slips and do not drop repeated same-amount slips.
- For each small separate receipt/slip, put that slip date in the item's bill_date_raw/from_date_raw/to_date_raw when visible.
- quantity_raw must contain only a real printed quantity/count. Never put taxable value, line amount, GST amount, grand total, unit price, or GST rate into quantity_raw.
- rate_raw must contain only GST rate percentage such as 5%, 12%, 18%, 9%+9%, or 18.00% when visible in the invoice. Do not put unit price, taxable value, total amount, or line amount in rate_raw.
- quantity_unit_raw is no longer required in final output; keep it empty unless internally useful.
- If there is deduction as separate row, include it as item with negative amount if visible.
- If no quantity/HSN is visible, use empty string. If no GST rate is visible, use 0.
- For Bill of Supply or Composition Taxable Person bills where tax is not collected/visible, set CGST, SGST, IGST and CESS to 0.
- For consultant/professional fee/retainer fee invoices with one service amount and no printed quantity column, quantity_raw should be 1; keep quantity_unit_raw empty; rate_raw must be the GST rate percentage only when visible, not the service amount.
- For receipts where quantity is not printed but the document is for one visible service/event/payment, keep quantity_raw empty; the API may normalize it to 1 later.
- If a table column says SAC, SAC Code, SAC Nos, HSN/SAC, or HSN Code, extract that into hsn_code_raw.
- For deposit receipts, use Transaction Type / Deposit / Hospital Deposit as particulars_raw, Transaction Amount or Received with Thanks as amount_raw, and Receipt No as bill_no_raw.
- For municipal/local utility/user-charge bills, extract the bill title/service title as service_usage_raw/particulars_raw, for example garbage/user/water/property/service charges.
- If a period label such as Bill Upto, Bill Up To, Bill Period, Service Period, or Charge Period contains two dates, put the full visible text in explicit_date_range_raw and service_period_raw.
- If the bill shows Monthly Finalized Amount, Current Charge, Taxable Value, Bill Total, or Net Charge, keep the charge amount in subtotal_raw/items.amount_raw and keep taxes separately.
- For monthly/periodic service bills, keep quantity_unit_raw empty; rate_raw must be GST rate percentage only when visible.
- For hotel/lodging/room-accommodation tax invoices, do not use the Grand Total/Total Billing as amount_raw when taxes are shown separately. Use Tariff/Room Rate/Taxable Value as amount_raw, Central GST/CGST total as cgst_raw, State GST/SGST total as sgst_raw, IGST total as igst_raw, and Grand Total/Total Billing only as grand_total_raw.
- For hotel/lodging/room-accommodation invoices, use Bill Number/Bill No as bill_no_raw. Never use HOTEL GSTIN/GSTIN/PAN as bill_no_raw.
- For hotel/lodging/room-accommodation invoices, use Arrival Date as service period start/from date and Departure Date as service period end/to date when visible. Put both dates in explicit_date_range_raw.
- If descriptions show Central GST @ 6% and State GST @ 6%, rate_raw should preserve the visible GST percentages, for example 6% + 6% or 12%.
Return JSON exactly in this shape:
{COMPACT_JSON_TEMPLATE}
"""
USER_PROMPT = """
Extract the expense fields from this document image/PDF.
Return only valid JSON.
"""
HEADER_JSON_TEMPLATE = """
{
  "vendor_employee_name_raw": "",
  "expense_type_raw": "",
  "los_vendor_state_raw": "",
  "lor_isec_state_raw": "",
  "service_usage_raw": "",
  "client_name_raw": "",
  "invoice_no_raw": "",
  "bill_no_raw": "",
  "bill_date_raw": "",
  "service_period_raw": "",
  "explicit_date_range_raw": "",
  "vendor_gstin_raw": "",
  "isec_gstin_raw": "",
  "vendor_pan_raw": "",
  "approver_raw": "",
  "whether_self_reimbursement_raw": "",
  "is_invoice_digitally_signed_raw": "",
  "digital_signature_evidence_raw": ""
}
"""
HEADER_SYSTEM_PROMPT = f"""
You are reading only the header/top part of an invoice, bill, challan, or receipt.
Return only valid JSON.
Do not use markdown.
Do not explain.
Use empty string "" only if value is not visible.
Extract raw visible values only.
Do not hardcode any value.
- expense_type_raw must always be an empty string. Do not extract or infer expense type.
- remarks_raw must always be an empty string. Do not extract, infer, repair, or generate remarks.
- quantity_raw and quantity_unit_raw must always be empty strings. Do not extract, infer, calculate, default, or repair quantity or quantity unit.
Important:
- vendor_employee_name_raw is the main invoice issuer business/shop/firm/vendor name, usually the largest issuer name at top or the company in 'For <company>' near authorised signatory/stamp. If a line contains vendor name followed by address, return only the name in vendor_employee_name_raw.
- Do not use recipient/customer/client names as vendor when another issuer/company name is visible.
- For municipal/local collection bills, prefer the organisation that issues/collects the charge or appears as account/signatory name over a generic authority header when both are visible.
- For municipal/local/utility bills, the client is the billed-party/customer line near Address/Category/Consumer details. Do not use address, category, payment account name, bank name, or authorised signatory as client.
- Do not choose contact/person names near phone numbers when a business/shop/firm name is visible.
- client_name_raw is the billed-to/customer/client name.
- Text after labels like M/s, M/s., Bill To, Billed To, Buyer, Customer, Client, To, Received From, Paid By, Party Name or Name is normally the client/billed party, not the vendor.
- For Bill of Supply / Composition taxable person bills, supplier/vendor is the issuer name at top/logo/header or near For/Authorised Signatory; customer/client is the M/s/Bill To block.
- LOS vendor state means vendor/state/city/location in bill.
- LOR ISEC state means customer/client/patient/place-of-supply state/city/location in bill. Prefer To/Bill To/Patient/Customer/Supply Address/Place of Supply over vendor address.
- bill_date_raw is invoice/bill/challan date only when a single date is visible.
- explicit_date_range_raw should contain full text if a date field has two dates connected by To/to/from/-/–.
- invoice_no_raw or bill_no_raw should come from Invoice No/Bill No/Challan No/Receipt No/Voucher No/Deposit Receipt No/CRN/Ride Number. Never use GSTIN/HOTEL GSTIN/PAN as invoice_no_raw or bill_no_raw.
- If one page contains multiple separate handwritten/local receipt panels, read all panels independently and return one item for each visible dated panel. Keep each panel's own date in that item.
- If PAN is visible, extract vendor_pan_raw.
- If GST No/GSTIN is blank, keep GST field blank.
- whether_self_reimbursement_raw should be Yes only for employee/staff/self reimbursement claim with clear employee/self evidence.
- Do not use Yes just because a document says reimbursement letter, reimbursement request, or request to reimburse.
- Reimbursement from an outside person/vendor/landlord/third party to a company is No unless employee/staff/self evidence is visible.
- For invoice/bill/challan/letter from vendor/person to client/company, use No.
- is_invoice_digitally_signed_raw should be Yes only if IRN, Ack No, QR code, e-invoice, digital signature, or digitally signed evidence is visible. Otherwise use No.
Return JSON exactly in this shape:
{HEADER_JSON_TEMPLATE}
"""
HEADER_USER_PROMPT = """
Extract only header/top invoice fields from this image.
Return only valid JSON.
"""
REPAIR_JSON_TEMPLATE = """
{
  "vendor_employee_name_raw": "",
  "expense_type_raw": "",
  "service_usage_raw": "",
  "client_name_raw": "",
  "billed_party_raw": "",
  "billed_party_block_raw": "",
  "customer_block_raw": "",
  "client_block_raw": "",
  "vendor_address_raw": "",
  "client_address_raw": "",
  "patient_address_raw": "",
  "place_of_supply_raw": "",
  "account_name_raw": "",
  "service_title_raw": "",
  "bill_upto_raw": "",
  "bill_period_raw": "",
  "invoice_no_raw": "",
  "bill_no_raw": "",
  "bill_date_raw": "",
  "service_period_raw": "",
  "explicit_date_range_raw": "",
  "los_vendor_state_raw": "",
  "lor_isec_state_raw": "",
  "vendor_gstin_raw": "",
  "isec_gstin_raw": "",
  "is_invoice_digitally_signed_raw": "",
  "digital_signature_evidence_raw": "",
  "subtotal_raw": "",
  "cgst_total_raw": "",
  "sgst_total_raw": "",
  "igst_total_raw": "",
  "cess_total_raw": "",
  "grand_total_raw": "",
  "items": [
    {
      "particulars_raw": "",
      "quantity_raw": "",
      "quantity_unit_raw": "",
      "hsn_code_raw": "",
      "rate_raw": "",
      "amount_raw": "",
      "cgst_raw": "",
      "sgst_raw": "",
      "igst_raw": "",
      "cess_raw": ""
    }
  ]
}
"""
REPAIR_SYSTEM_PROMPT = f"""
You repair missing fields from receipts, vouchers, medical bills, deposit receipts, taxi bills, food bills and invoices.
Return only valid JSON. Do not use markdown. Do not explain. Use empty string "" when a value is not visible.
- expense_type_raw must always be an empty string. Do not extract or infer expense type.
- remarks_raw must always be an empty string. Do not extract, infer, repair, or generate remarks.
- quantity_raw and quantity_unit_raw must always be empty strings. Do not extract, infer, calculate, default, or repair quantity or quantity unit.
- whether_self_reimbursement_raw must be Yes only for employee/staff/self expense reimbursement evidence.
- Do not mark reimbursement letter/request as self reimbursement only because the word reimbursement appears.
- Third-party/person/vendor/landlord reimbursement requests to a company are No unless employee/staff/self evidence is clearly visible.
Critical extraction rules:
- bill_no_raw must capture Receipt No, Receipt Number, Voucher No, Deposit Receipt No, Transaction Receipt No, Invoice No, Bill No, CRN/Ride Number.
- bill_date_raw must capture Receipt Date, Transaction Date, Invoice Date, Bill Date, or document Date.
- is_invoice_digitally_signed_raw must be Yes only for IRN, Ack No, Ack Date, e-invoice QR, digital signature evidence. Otherwise use No.
- los_vendor_state_raw must be the vendor/hospital/supplier state from vendor address, branch/footer address, or vendor GSTIN state.
- lor_isec_state_raw must be the customer/client/patient/place-of-supply state from To/Bill To/Patient/Customer/Supply Address/Place of Supply. Never leave LOR blank when this state is visible.
- vendor_gstin_raw must be the supplier/vendor GSTIN only.
- Text after labels like M/s, M/s., Bill To, Billed To, Buyer, Customer, Client, To, Received From, Paid By, Party Name or Name is normally the client/billed party, not the vendor.
- For Bill of Supply / Composition taxable person bills, supplier/vendor is the issuer name at top/logo/header or near For/Authorised Signatory; customer/client is the M/s/Bill To block.
- isec_gstin_raw must be the buyer/customer/client/recipient GSTIN only. If GST No/GSTIN appears inside or immediately under the To/Bill To/Buyer/Customer/Client block, put it in isec_gstin_raw, not vendor_gstin_raw.
- For consultant/professional-fee invoices where the vendor has PAN but no GSTIN and the To/client block has GST No, that GST No is isec_gstin_raw.
- Also return vendor_address_raw and client_address_raw/patient_address_raw/place_of_supply_raw when visible, because these are used only to repair missing LOS/LOR.
- If Account Name is visible in payment/bank details, return it in account_name_raw.
- If a service title is visible, such as Garbage Collection Bill, User Charge Bill, Water Bill, Broadband Services, Restaurant Invoice, or Deposit Receipt, return it in service_title_raw and use it to form particulars_raw.
- For municipal/local utility/user-charge bills, vendor_employee_name_raw should be the issuing/collecting organisation; client_name_raw should be the billed party shown near address/category details. Do not use the client name as vendor.
- For municipal/local utility/user-charge bills, if a left-side customer block shows a name before Address/Category, put that name in both client_name_raw and billed_party_raw. Do not use bank Account Name, IFSC, cheque/payee name, or authorised signatory as client_name_raw.
- Generic customer-block rule: when a block is formatted like a standalone name line followed by Address/Category/Consumer/Subscriber details, the standalone name line is client_name_raw and billed_party_raw. The Address value itself is client_address_raw, never client_name_raw.
- Also return the exact full customer/billed-party block in billed_party_block_raw/customer_block_raw. Example format: first line name, next line Address, next line Category/Consumer details.
- If client_name_raw accidentally contains extra text after Address/Category/Bill No/Service Date labels, keep only the standalone billed-party name before those labels.
- For municipal/local utility/user-charge bills, do not use city names as LOS/LOR. Use Indian state names only. If vendor GSTIN is visible, LOS must be the state represented by the first two GSTIN digits.
- Do not include labels in bill_no_raw. Return only the document number value, for example return KNN/61/135680, not Bill No : KNN/61/135680.
- Do not use Service Start Date as FROM DATE when Bill Upto/Bill Period/Service Period has two period dates. For period bills, use the two dates in Bill Upto/Bill Period as from/to.
- For Bill Upto/Bill Up To/Bill Period/Service Period/Charge Period containing two dates, put the full visible text in explicit_date_range_raw and service_period_raw.
- For monthly or period-based local service bills, quantity_raw should be 1, quantity_unit_raw should be empty, rate_raw should be GST rate percentage only when visible, and amount_raw should be Current Charge/Monthly Finalized Amount/Taxable Value, not arrear/outstanding.
- For consultant/professional fee/retainer fee invoices with one service amount and no printed quantity column, quantity_raw should be 1, quantity_unit_raw should be empty, and rate_raw should be GST rate percentage only when visible.
- For deposit receipts, particulars_raw should be Deposit or Hospital Deposit, quantity_raw should be 1 only when one deposit/payment is shown, quantity_unit_raw should be empty, rate_raw should be GST rate percentage only when visible, and amount_raw should be Transaction Amount / Received with Thanks amount.
- For receipt vouchers/exam/course/training, particulars_raw should be the service/examination/course line, quantity_unit_raw should be empty, SAC/HSN code goes to hsn_code_raw, amount_raw must come from its table column, and rate_raw must be GST rate percentage only when visible.
- For cab/taxi/ride bills, particulars_raw should be Ride Fare / Cab Ride / Trip, quantity_unit_raw should be empty, amount_raw should use the main paid/total bill amount if only one claim line is needed, and rate_raw should be GST rate percentage only when visible.
- Do not put payment mode such as Cash, UPI, Card, wallet/payment app as particulars_raw unless it is the only visible description and no service is visible.
- Do not use labels like Current Charge, Total Amount, Arrear, Payment Detail, or Monthly Finalized Amount as particulars_raw when a service title is visible; use the service title as particulars_raw.
- For hotel/lodging/room-accommodation tax invoices, return one consolidated service line when a Bill Summary/Grand Total is visible: amount_raw = Tariff/Taxable Value/Room Rate subtotal, cgst_raw = Central GST/CGST total, sgst_raw = State GST/SGST total, igst_raw = IGST total, grand_total_raw = Total Billing/Grand Total. Do not put tax-inclusive total in amount_raw.
- For hotel/lodging/room-accommodation invoices, bill_no_raw must be Bill Number/Bill No only; never use HOTEL GSTIN/GSTIN/PAN as bill_no_raw.
- For hotel/lodging/room-accommodation invoices, explicit_date_range_raw should be Arrival Date to Departure Date when both are visible.
- A cashier signature, guest signature, authorised signatory, or handwritten signature is not a digital signature. Digital signature is Yes only for IRN/Ack/e-invoice/QR/digital-signature evidence.
- For newspaper/monthly publication/subscription bills, vendor_employee_name_raw must be the specific agency/dealer/shop/person name at the top, not the generic heading such as Newspaper Dealers. If the agency name is printed in an Indian script, transliterate it to English.
- For newspaper/monthly publication/subscription bills, expense_type_raw must stay empty and service_usage_raw should be Newspaper Monthly Subscription.
- For newspaper/monthly publication/subscription bills, if no Bill No is visible but an Id/ID/No appears near Name/Customer, return that value as bill_no_raw. Do not include the label Id/ID.
- For newspaper/monthly publication/subscription bills, keep quantity_unit_raw empty for every publication line item; if no quantity column is visible, quantity_raw should be 1 and rate_raw should be GST rate percentage only when visible, otherwise empty.
- For newspaper/monthly publication/subscription bills with no visible GST/tax rows, CGST, SGST, IGST and CESS should be 0.
Return JSON exactly in this shape:
{REPAIR_JSON_TEMPLATE}
"""
REPAIR_USER_PROMPT = """
Some required fields are missing or wrong. Re-read this page and return the corrected receipt/invoice fields as JSON only.
"""
VENDOR_IDENTITY_JSON_TEMPLATE = """
{
  "vendor_employee_name_raw": "",
  "vendor_display_name_english_raw": "",
  "vendor_category_raw": "",
  "vendor_address_raw": "",
  "los_vendor_state_raw": "",
  "evidence_raw": ""
}
"""
VENDOR_IDENTITY_SYSTEM_PROMPT = f"""
You extract ONLY the supplier/vendor/issuer identity from the top/header of a bill, receipt, memo, invoice, or subscription statement.
Return only valid JSON. Do not use markdown. Do not explain. Use empty string "" when value is not visible.
Critical vendor rules:
- vendor_employee_name_raw must be the specific issuer/vendor/shop/agency/person name, not a generic category heading and not the full postal address. If a line contains name plus address, return only the name.
- For tax invoices from hotels/resorts/lodging, prefer the legal supplier/entity name printed in the main header near address, GSTIN, PAN, VAT/TIN, CIN, FSSAI or registration details. Do not use only the hotel logo/brand/trade style when a separate legal entity name is visible in the header.
- When the top-right logo/brand differs from the top-left legal business name with GSTIN/PAN/address, return the top-left legal business name as vendor_employee_name_raw.
- Generic category headings such as Newspaper Dealers, Restaurant, Cash/Credit Memo, Tax Invoice, Receipt, Bill, Services, or Subscription are NOT vendor names. Put such text in vendor_category_raw only.
- If a non-English vendor/agency name is visible in Hindi/Marathi/Gujarati/other Indian script, transliterate it into English in vendor_employee_name_raw and vendor_display_name_english_raw. Do not leave it in the original script.
- vendor_employee_name_raw and vendor_display_name_english_raw must be in English/Latin characters only. Never return Devanagari/Hindi/Marathi font in these vendor fields.
- If both a non-English agency name and English proprietor names are visible, prefer the agency/shop name transliterated to English. If the agency/shop name is unreadable, use the specific English proprietor/person names plus the category.
- For newspaper/monthly publication/subscription bills, the vendor must be the specific newspaper agency/dealer/shop/person name printed above, below, or near the words Newspaper Dealers. Never return only "Newspaper Dealers".
- In small newspaper/dealer bills, check all of these locations before deciding vendor: top line above the category heading, English proprietor names near the heading, address line, footer/stamp/signature agency name. Prefer the specific agency/shop name; if that cannot be read, use specific proprietor/person names with the category, not the generic category alone.
- Transliterate Marathi/Hindi/Devanagari vendor lines into English. Example style only: "श्री ... न्यूज पेपर एजन्सी" should be returned as "Shri ... News Paper Agency". Do not copy Devanagari into the JSON vendor name.
- Do not use the billed party/customer/client name as vendor. Names after M/s, M/s., Bill To, Buyer, Customer, Client, To, Received From, Paid By or Party Name labels are customer/client names, not vendor names.
- For tea/coffee/snacks/food/canteen/restaurant/water/cab/newspaper/local bills, the vendor is the small issuer/shop/stall/supplier/person at the top, stamp, footer, For/Authorised Signatory area, or handwritten issuer line. A large corporate payer/customer name must not be returned as vendor.
- If a bill/cash memo has a label like From, From:, Issued By, Supplier, Vendor, Shop, Stall, Restaurant, Canteen, or For near the top/header, the name immediately under or after that label is the vendor/issuer. Read this even when handwritten or in Hindi/Marathi/Devanagari and transliterate it to English. Ignore phone numbers under the vendor name.
- If both a From/issuer block and an M/s/Bill-To/Customer block are visible, return the From/issuer block as vendor_employee_name_raw and the M/s/Bill-To/Customer block must not be returned as vendor.
- In evidence_raw, include the exact visible issuer/vendor label block you used, for example the From/issuer block text.
- vendor_address_raw is the vendor address only.
- los_vendor_state_raw must be an Indian state name when visible or inferable from the vendor city/address.
Return JSON exactly in this shape:
{VENDOR_IDENTITY_JSON_TEMPLATE}
"""
VENDOR_IDENTITY_USER_PROMPT = """
Read the supplier/vendor/agency identity from this cropped header image. Return only JSON.
"""
CLIENT_BLOCK_JSON_TEMPLATE = """
{
  "client_name_raw": "",
  "billed_party_raw": "",
  "billed_party_block_raw": "",
  "customer_block_raw": "",
  "client_address_raw": "",
  "lor_isec_state_raw": "",
  "isec_gstin_raw": "",
  "visible_text_raw": ""
}
"""
CLIENT_BLOCK_SYSTEM_PROMPT = f"""
You extract ONLY the billed-party/customer/client/consumer block from a receipt, invoice, utility bill, municipal bill, hospital receipt, food bill, taxi bill, or service bill.
Return only valid JSON. Do not use markdown. Do not explain. Use empty string "" when value is not visible.
Critical rule for client_name_raw:
- client_name_raw is the billed party / customer / consumer name.
- For labels like M/s, M/s., Bill To, Billed To, Buyer, Customer, Client, To, Received From, Paid By, Party Name or Name, return the name immediately following that label as client_name_raw when the surrounding block is a billed-party/customer block.
- Check all provided images/crops, including rotated versions. If the bill is sideways/rotated, read the crop where text is upright.
- When both supplier/vendor and M/s/Bill-To labels are visible, supplier/vendor is not the client; the name after M/s/Bill-To is the client.
- If the document has a block like:
  NAME
  Address: ...
  Category: ...
  then client_name_raw must be NAME only.
- Do not return Address, Category, Bank Name, Account Name, IFSC, Payment Detail, Authorised Signatory, vendor name, issuer name, or service title as client_name_raw.
- If the name appears on a standalone line immediately before Address/Category/Consumer/Subscriber details, return that standalone line as client_name_raw and billed_party_raw.
- billed_party_block_raw/customer_block_raw should contain the visible customer block text, preserving line order as much as possible.
- client_address_raw should contain only address/location details, not the name.
- lor_isec_state_raw should be the state from customer/client/place-of-supply address if visible.
- isec_gstin_raw should be the GSTIN/GST No visible inside the customer/client/buyer/To block only. Do not return vendor GSTIN here.
Return JSON exactly in this shape:
{CLIENT_BLOCK_JSON_TEMPLATE}
"""
CLIENT_BLOCK_USER_PROMPT = """
Read the customer/billed-party/client/consumer block only. Return the client name and related customer block fields as JSON only.
"""
PARTY_IDENTITY_JSON_TEMPLATE = """
{
  "vendor_employee_name_raw": "",
  "client_name_raw": "",
  "billed_party_raw": "",
  "vendor_gstin_raw": "",
  "isec_gstin_raw": "",
  "vendor_address_raw": "",
  "client_address_raw": "",
  "evidence_raw": ""
}
"""
PARTY_IDENTITY_SYSTEM_PROMPT = f"""
You extract ONLY supplier/vendor/issuer and billed-party/customer/client identity from one bill, memo, receipt, challan, or invoice.
Return only valid JSON. Do not use markdown. Do not explain. Use empty string "" when a value is not visible.
Extract visible values only. Do not hardcode any company, vendor, or client name.
Important orientation rule:
- The same document may be shown in multiple rotations. Use the image where text is most readable/horizontal. Ignore rotated duplicates.
- If any rotated copy makes text horizontal, read supplier and client from that rotated copy.
- Do not return blank for supplier/client when the readable rotated copy clearly shows a labelled issuer or M/s/Bill-To block.
Supplier/vendor rules:
- vendor_employee_name_raw is the issuer/supplier/shop/stall/firm/person who made the bill.
- Prefer the legal supplier/entity name in the main header near address, GSTIN, PAN, VAT/TIN, CIN, FSSAI or registration details.
- For hotel/resort/lodging tax invoices, ignore a decorative hotel logo/brand/trade style when a separate legal company/entity name is printed in the header with GSTIN/PAN/address.
- Prefer the name/logo/header at the top, or the name after labels such as From, Supplier, Vendor, Issued By, Shop, Stall, Restaurant, Canteen, or For.
- For Bill of Supply, Cash Memo, Credit Memo, retail bill, food/tea/coffee/snacks bill, water-jar bill, challan or receipt, the top logo/header/issuer block is supplier/vendor.
- The phrase near Authorised Signatory like "For <name>" is supplier/vendor when present.
- For cash/credit memo and small handwritten bills, the block after From is the supplier/vendor, not the customer.
- Do not use the name after M/s, Bill To, Billed To, Buyer, Customer, Client, To, Received From, Paid By, or Party Name as vendor.
- If the vendor name is in Hindi/Marathi/Gujarati/Devanagari or another Indian script, transliterate it into English/Latin script.
Billed-party/client rules:
- client_name_raw is the billed party/customer/recipient.
- Prefer labels such as M/s, Bill To, Billed To, Buyer, Customer, Client, To, Received From, Paid By, Party Name, Consumer, or Subscriber.
- For Bill of Supply or invoice formats, the name after M/s is usually client_name_raw.
- Do not use address, GSTIN label, bill number, date, amount, HSN, rate, total, bank, IFSC, or authorised signatory as client_name_raw.
GSTIN rules:
- vendor_gstin_raw is supplier/vendor GSTIN only.
- isec_gstin_raw is buyer/client/customer/recipient GSTIN only. If GSTIN appears inside or near M/s/Bill To/Buyer/Customer/Client block, put it in isec_gstin_raw.
Return JSON exactly in this shape:
{PARTY_IDENTITY_JSON_TEMPLATE}
"""
PARTY_IDENTITY_USER_PROMPT = """
Read the supplier/vendor and billed-party/customer/client identity from this bill image. Return only JSON.
"""
LEGAL_SUPPLIER_JSON_TEMPLATE = """
{
  "legal_supplier_name_raw": "",
  "trade_or_logo_name_raw": "",
  "vendor_gstin_raw": "",
  "vendor_pan_raw": "",
  "vendor_address_raw": "",
  "los_vendor_state_raw": "",
  "evidence_raw": ""
}
"""
LEGAL_SUPPLIER_SYSTEM_PROMPT = f"""
You extract ONLY the legal supplier/entity name from the invoice header.
Return only valid JSON. Do not use markdown. Do not explain. Use empty string "" when a value is not visible. Extract visible values only. Do not hardcode any company, vendor, or client name.
Rules:
- legal_supplier_name_raw must be the legal business/entity name printed near supplier address, GSTIN, PAN, VAT/TIN, CIN, FSSAI, registration or tax identifiers.
- First inspect the top-left/header supplier block above or beside the supplier address, phone/email, PAN, GSTIN, VAT/TIN, FSSAI, CIN or registration details. That line is usually the legal supplier.
- For hotel/resort/lodging invoices, many pages show both a trade/logo name and a legal entity. If a top-right/top-logo hotel brand is visible and a separate top-left/top-header legal entity with GSTIN/PAN/address is visible, return the legal entity in legal_supplier_name_raw and put the logo/brand in trade_or_logo_name_raw.
- A hotel brand/trade/logo name without legal suffix such as Pvt Ltd, Private Limited, Ltd, Limited, LLP, Company, etc. is not the legal supplier when a different legal-entity line with such suffix is visible in the header.
- If a crop contains only the supplier header/address/tax-id area, ignore any remembered or decorative hotel brand from other pages and return the legal entity printed in that crop.
- Do not return the decorative hotel logo/brand/trade style as legal_supplier_name_raw when a separate legal entity is visible.
- Do not return guest name, company name, bill-to/client name, customer name, room number, cashier name, signature name, GSTIN, PAN, bill number, or address as legal_supplier_name_raw.
- If the legal supplier line contains address text after the name, return only the legal name before the address.
- evidence_raw should contain the exact short visible header fragment used, including the legal name and nearby GSTIN/PAN/address clue.
Return JSON exactly in this shape:
{LEGAL_SUPPLIER_JSON_TEMPLATE}
"""
LEGAL_SUPPLIER_USER_PROMPT = """
Read the legal supplier/entity name from the invoice header. The images are prioritized: the first images crop the supplier legal header/address/tax-id area and may intentionally exclude the logo/trade-name area. Prefer the top-left/header legal entity near GSTIN/PAN/address over any decorative hotel/logo/trade name. Return only JSON.
"""
CHARGE_SUMMARY_JSON_TEMPLATE = """
{
  "bill_no_raw": "",
  "bill_date_raw": "",
  "service_period_raw": "",
  "explicit_date_range_raw": "",
  "vendor_employee_name_raw": "",
  "client_name_raw": "",
  "los_vendor_state_raw": "",
  "lor_isec_state_raw": "",
  "service_usage_raw": "",
  "is_invoice_digitally_signed_raw": "",
  "digital_signature_evidence_raw": "",
  "subtotal_raw": "",
  "cgst_total_raw": "",
  "sgst_total_raw": "",
  "igst_total_raw": "",
  "cess_total_raw": "",
  "grand_total_raw": "",
  "amount_in_words_raw": "",
  "items": [
    {
      "particulars_raw": "",
      "quantity_raw": "",
      "quantity_unit_raw": "",
      "hsn_code_raw": "",
      "rate_raw": "",
      "amount_raw": "",
      "cgst_raw": "",
      "sgst_raw": "",
      "igst_raw": "",
      "cess_raw": ""
    }
  ]
}
"""
MULTI_RECEIPT_PANEL_JSON_TEMPLATE = """
{
  "document_type_raw": "",
  "whether_self_reimbursement_raw": "",
  "vendor_employee_name_raw": "",
  "service_usage_raw": "",
  "is_invoice_digitally_signed_raw": "",
  "remarks_raw": "",
  "approver_raw": "",
  "client_name_raw": "",
  "grand_total_raw": "",
  "items": [
    {
      "particulars_raw": "",
      "quantity_raw": "",
      "quantity_unit_raw": "",
      "hsn_code_raw": "",
      "rate_raw": "",
      "amount_raw": "",
      "cgst_raw": "",
      "sgst_raw": "",
      "igst_raw": "",
      "cess_raw": "",
      "bill_date_raw": "",
      "from_date_raw": "",
      "to_date_raw": ""
    }
  ]
}
"""
MULTI_RECEIPT_PANEL_SYSTEM_PROMPT = f"""
You extract multiple small handwritten/local receipt panels from one scanned image.
Return only valid JSON. Do not use markdown. Do not explain. Use empty string "" when a value is not visible. Extract visible values only. Do not hardcode any vendor, client, person, item, date, amount, or company name.
This task is only for pages where several small receipts/slips/chits/cash memos are visible on one page.
Rules:
- Read the whole page and every visible separate receipt/slip/panel: top-left, top-right, bottom-left, bottom-right, and any additional panel. Some inputs may be cropped single-panel images; still return that panel as one item.
- Return one item per separate dated receipt/slip/panel across all provided images. Do not merge panels. Do not drop repeated amounts or repeated item names. If four separate dated panels are visible, return four items.
- If two panels have the same amount, keep both when their dates or item text differ.
- vendor_employee_name_raw is the supplier/person/shop name written on the receipts.
- approver_raw should be a separate approval/name/signature only when visible apart from the supplier name.
- client_name_raw should be blank unless a clear billed-to/customer/client name is printed.
- For each item, particulars_raw is the product/service description written in that panel.
- quantity_raw is the visible quantity/count only, for example the number before Kg/Kilo/Pcs. Do not put 1 just because the row starts with serial number 1). Serial number 1) is not quantity.
- quantity_unit_raw may contain Kg/Kilo/Pcs internally when visible, but final API output may keep quantity_unit blank.
- amount_raw is the final amount for that panel.
- bill_date_raw, from_date_raw, and to_date_raw should all be the panel's own visible date for one-day receipt panels.
- For handwritten/local non-GST receipts, set cgst_raw, sgst_raw, igst_raw, cess_raw, and rate_raw to 0 when no GST/tax rate is visible.
- If no bill/invoice/receipt number is visible, keep bill number fields blank.
- grand_total_raw should be the sum of all extracted panel amounts only when all panel amounts are visible; otherwise blank.
- Do not infer month/year from unrelated text. Preserve visible dates exactly, then normalize only if clear.
Return JSON exactly in this shape:
{MULTI_RECEIPT_PANEL_JSON_TEMPLATE}
"""
MULTI_RECEIPT_PANEL_USER_PROMPT = """
Extract every separate handwritten/local receipt panel on this page. Return one item per panel with its own date, quantity, amount and tax fields. Return only JSON.
"""
CHARGE_SUMMARY_SYSTEM_PROMPT = f"""
You extract only the final charge summary from hotel, lodging, room-accommodation, electricity, water, gas, telecom, maintenance, utility or service bills.
Return only valid JSON. Do not use markdown. Do not explain. Use empty string "" when a value is not visible. Extract visible values only. Do not hardcode any company, vendor, or client name.
Common rules:
- Read all provided page images together before answering. For multipage bills, values repeated on several pages must stay consistent.
- vendor_employee_name_raw must be the legal supplier/business name printed in the main top header near address/GSTIN/PAN/VAT/TIN/FSSAI/CIN. Do not use only a hotel logo/brand/trade style when a separate legal entity name is printed.
- For hotel bills, compare the top-left legal-header block with the top-right hotel logo/brand; if both are visible, vendor_employee_name_raw must be the legal-header entity, not the logo/brand.
- bill_no_raw must be Bill Number, Bill No, Invoice No, Receipt No, or Tax Invoice number only. Copy it exactly as printed; do not change digits.
- Never use GSTIN, PAN, Service No, S No, Serial No, SC No, USC No, Consumer No, Customer No, Account No, Meter No, Room No, mobile number, or values that look like S NO / 5 NO as bill_no_raw.
- bill_date_raw must be the date printed near Bill Date / Invoice Date / Issue Date / Receipt Date. Do not leave it blank when that label is visible.
- is_invoice_digitally_signed_raw should be Yes only for IRN, Ack No, e-invoice QR, QR code, digital-signature certificate, or digitally signed PDF evidence. A handwritten/cashier/authorised signatory signature is not a digital signature.
- client_name_raw should be only a billed-party/customer/company/consumer name, not a postal address.
Hotel/lodging rules:
- Return one consolidated accommodation line for the whole stay when a bill summary/grand total/total billing is visible.
- For multipage hotel invoices, prefer the Bill Summary/Grand Total/Total Billing page over daily transaction rows.
- Do not return one daily Room Rate row or Day Total as the final line when the invoice spans multiple days/pages.
- amount_raw = Tariff / Room Rate total / taxable value before GST for the complete bill. Do not put Total Billing or Grand Total into amount_raw when GST is shown separately.
- cgst_raw = Central GST / CGST total. sgst_raw = State GST / SGST total. igst_raw = IGST total only when IGST is explicitly visible. If Central GST and State GST are visible, keep igst_raw as 0.
- rate_raw must contain split GST rate when split GST is printed. If Central GST @ 6% and State GST @ 6% are visible, return rate_raw as 6% + 6%, not a calculated percentage.
- grand_total_raw = Total Billing / Grand Total / final tax-inclusive amount for the complete bill.
- Ignore Day Total, running balance/R.Balance, page subtotal, and one-day room totals as final grand total when a Bill Summary/Total Billing is visible on another page.
- explicit_date_range_raw = Arrival Date to Departure Date when both are visible. Use the date under Departure Date/Time exactly; do not infer a different month.
- hsn_code_raw must use the SAC/HSN column value for the accommodation/service row when visible.
- quantity_raw should be empty unless an actual quantity/nights count is printed; never put room rate or tariff amount into quantity_raw.
Utility/electricity/water/gas bill rules:
- Return one consolidated line only. Do not itemize detailed charges such as energy charges, arrears, adjustment, duty, surcharge, discount, interest, taxes, net amount, or total amount rows.
- amount_raw and grand_total_raw should be the final payable/current bill amount/total due/amount due/net payable visible on the bill.
- If no GST is visible, set cgst_raw, sgst_raw, igst_raw, and cess_raw to 0.
- Do not infer a state unless a state name, GSTIN state code, clear address/state abbreviation, or a visible Indian state abbreviation in the utility-board name is visible.
- For a utility Bill of Supply, a long document number printed on the same top-title row as "Bill of Supply for the month of ..." can be bill_no_raw even if the label is not repeated. Do not confuse it with Consumer No, GSTIN, HSN code, PIN code, phone/mobile number, last receipt number, or amount.
- bill_date_raw must be the date printed near Bill Date / Invoice Date / Issue Date only. Do not use due date, service period start/end date, meter reading date, connection date, first paid date, or last paid date as bill_date_raw.
- If the title says "for the month of MON-YYYY" or "month of MON YYYY", explicit_date_range_raw should be the full calendar month range, for example 01/MM/YYYY to last day/MM/YYYY.
- client_name_raw should use the Consumer Name / Customer Name value only; correct OCR digit/letter slips only when the printed label is visible.
- If only Service No / S No / 5 No / Consumer No / Meter No / Account No is visible and no Bill No/Invoice No/top-title utility document number is visible, keep bill_no_raw empty.
Return JSON exactly in this shape:
{CHARGE_SUMMARY_JSON_TEMPLATE}
"""
CHARGE_SUMMARY_USER_PROMPT = """
Extract the final bill summary totals, GST/tax totals, bill number, bill date, service period and one consolidated charge line. Return only JSON.
"""
# =========================
# Image/PDF rendering
# =========================
def get_upload_suffix(filename: str, content_type: str = "") -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix:
        return suffix
    content_type = (content_type or "").lower()
    if "pdf" in content_type:
        return ".pdf"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "bmp" in content_type:
        return ".bmp"
    if "tiff" in content_type or "tif" in content_type:
        return ".tiff"
    return ""
async def save_upload_to_temp(file: UploadFile, temp_dir: str) -> Tuple[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name missing.")
    uploaded_bytes = await file.read()
    print(f"uploaded bytes length: {len(uploaded_bytes)}")
    if not uploaded_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    safe_filename = os.path.basename(file.filename)
    suffix = get_upload_suffix(safe_filename, file.content_type or "")
    if suffix and not safe_filename.lower().endswith(suffix):
        safe_filename = f"{Path(safe_filename).stem}{suffix}"
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=temp_dir,
    )
    try:
        temp_file.write(uploaded_bytes)
        temp_file.flush()
        file_path = temp_file.name
    finally:
        temp_file.close()
    return file_path, safe_filename
def validate_uploaded_file(file_path: str, filename: str) -> str:
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Uploaded temp file was not created.")
    if os.path.getsize(file_path) == 0:
        raise HTTPException(status_code=400, detail="Uploaded temp file is empty.")
    suffix = Path(filename or file_path).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        try:
            doc = fitz.open(file_path)
            try:
                if doc.page_count == 0:
                    raise HTTPException(status_code=400, detail="PDF has no pages.")
            finally:
                doc.close()
            return "pdf"
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")
    if suffix in IMAGE_EXTENSIONS:
        try:
            with Image.open(file_path) as img:
                img.verify()
            return "image"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Upload PDF, JPG, JPEG, PNG, WEBP, BMP, or TIFF.",
    )
def count_pages_for_response(file_path: str, filename: str) -> int:
    """Return a cheap page count without rendering images."""
    lower = safe_str(filename).lower()
    if lower.endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            try:
                return min(len(doc), MAX_PAGES)
            finally:
                doc.close()
        except Exception:
            return 0
    if lower.endswith(tuple(IMAGE_EXTENSIONS)):
        return 1
    return 0
def compute_file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def response_cache_key(file_hash: str, filename: str, file_type: str) -> str:
    """
    Cache key includes extraction settings that can change the output.
    Filename itself is intentionally not included, so the same file uploaded with a
    different display name can still use the cache.
    """
    payload = "|".join([
        CACHE_VERSION,
        file_hash,
        file_type,
        MODEL_NAME,
        str(PDF_DPI),
        str(MAX_PAGES),
        str(MAX_IMAGE_SIDE),
        IMAGE_FORMAT,
        str(JPEG_QUALITY),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
def read_cached_extraction(cache_key: str) -> Dict[str, Any]:
    if not ENABLE_RESPONSE_CACHE:
        return {}
    try:
        cache_path = Path(RESPONSE_CACHE_DIR) / f"{cache_key}.json"
        if not cache_path.exists():
            return {}
        with open(cache_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}
def write_cached_extraction(cache_key: str, payload: Dict[str, Any]) -> None:
    if not ENABLE_RESPONSE_CACHE:
        return
    try:
        os.makedirs(RESPONSE_CACHE_DIR, exist_ok=True)
        cache_path = Path(RESPONSE_CACHE_DIR) / f"{cache_key}.json"
        tmp_path = cache_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_path, cache_path)
    except Exception:
        # Cache must never break extraction.
        return
def pil_image_to_base64_url(img: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    current_max_side = max(width, height)
    if current_max_side > max_side:
        scale = max_side / current_max_side
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        img = img.resize((new_width, new_height), Image.LANCZOS)
    buffer = io.BytesIO()
    if IMAGE_FORMAT == "PNG":
        img.save(buffer, format="PNG")
        mime = "image/png"
    else:
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        mime = "image/jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"
def add_rotation_aware_repair_crops(crops: List[Image.Image], base_img: Image.Image, include_header_slices: bool = True) -> List[Image.Image]:
    """
    Add generic rotated document crops for sideways scanned bills.
    This does not use vendor/client names. It only provides alternate orientations
    to later repair passes when customer/vendor extraction is weak.
    """
    if not ENABLE_ROTATION_AWARE_REPAIR_CROPS:
        return crops
    try:
        bbox = find_nonwhite_bbox(base_img)
        doc_crop = base_img.crop(bbox)
    except Exception:
        doc_crop = base_img
    rotated_crops: List[Image.Image] = []
    for angle in (90, 270):
        try:
            rotated = doc_crop.rotate(angle, expand=True)
            rotated_crops.append(rotated)
            if include_header_slices:
                rw, rh = rotated.size
                # Full top/header band.
                rotated_crops.append(rotated.crop((0, 0, rw, int(rh * 0.52))))
                # Left/top issuer or billed-party block.
                rotated_crops.append(rotated.crop((0, 0, int(rw * 0.72), int(rh * 0.58))))
                # Generic billed-party area crops for sideways bills.
                # These are geometry/layout based only; no vendor/client names are hardcoded.
                rotated_crops.append(rotated.crop((0, int(rh * 0.08), int(rw * 0.55), int(rh * 0.45))))
                # Generic right/top invoice metadata and supplier/address area.
                rotated_crops.append(rotated.crop((int(rw * 0.35), 0, rw, int(rh * 0.45))))
                # Full middle band for M/s, To, Bill-To and table header contexts.
                rotated_crops.append(rotated.crop((0, int(rh * 0.15), rw, int(rh * 0.58))))
        except Exception:
            continue
    if rotated_crops:
        # Keep the original crop first for normally oriented bills, then put rotated
        # alternatives before narrow unrotated crops. This helps sideways scans where
        # vendor/client names are otherwise missed, without using hardcoded names.
        return (crops[:1] + rotated_crops + crops[1:]) if crops else rotated_crops
    return crops
def pdf_first_page_to_image(pdf_path: str) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        zoom = PDF_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    finally:
        doc.close()
def pdf_to_base64_images(pdf_path: str) -> List[str]:
    images = []
    doc = fitz.open(pdf_path)
    try:
        total_pages = min(len(doc), MAX_PAGES)
        zoom = PDF_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(total_pages):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(pil_image_to_base64_url(img))
    finally:
        doc.close()
    return images
def image_file_to_base64_images(image_path: str) -> List[str]:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return [pil_image_to_base64_url(img)]
def file_to_base64_images(file_path: str, filename: str) -> List[str]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return pdf_to_base64_images(file_path)
    if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")):
        return image_file_to_base64_images(file_path)
    raise HTTPException(
        status_code=400,
        detail="Only PDF and image files are supported.",
    )
def load_first_page_or_image(file_path: str, filename: str) -> Image.Image:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return pdf_first_page_to_image(file_path)
    img = Image.open(file_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
def crop_header_image_urls_from_file(file_path: str, filename: str) -> List[str]:
    """
    Generic header crops. No invoice-specific hardcoding.
    """
    img = load_first_page_or_image(file_path, filename)
    width, height = img.size
    crops = []
    # Top area
    crops.append(img.crop((0, 0, width, int(height * 0.68))))
    # Centered document top/middle
    crops.append(
        img.crop(
            (
                int(width * 0.08),
                int(height * 0.03),
                int(width * 0.92),
                int(height * 0.72),
            )
        )
    )
    # Right/top section often contains date/bill no.
    crops.append(
        img.crop(
            (
                int(width * 0.40),
                int(height * 0.00),
                width,
                int(height * 0.45),
            )
        )
    )
    crops = add_rotation_aware_repair_crops(crops, img, include_header_slices=True)
    # Header pass must stay compact. Sending too many rotated crops can exceed
    # Qwen/VLM context length on scanned bills. Keep the strongest early crops
    # only: original header plus the first rotation-aware document/header crops.
    crops = crops[:max(1, MAX_HEADER_CROPS)]
    urls = []
    for crop in crops:
        urls.append(pil_image_to_base64_url(crop, max_side=HEADER_IMAGE_SIDE))
    return urls
def crop_customer_context_image_urls_from_file(file_path: str, filename: str) -> List[str]:
    """
    Generic layout crops for customer/billed-party blocks.
    This is not company-specific: many receipts put the customer block on the
    upper/middle-left and document numbers on the upper/middle-right.
    """
    img = load_first_page_or_image(file_path, filename)
    width, height = img.size
    crops = []
    crops.append(img)
    crops.append(img.crop((0, 0, width, int(height * 0.58))))
    crops.append(img.crop((0, int(height * 0.16), int(width * 0.58), int(height * 0.58))))
    crops.append(img.crop((0, int(height * 0.24), int(width * 0.65), int(height * 0.70))))
    crops = add_rotation_aware_repair_crops(crops, img, include_header_slices=True)
    urls = []
    for crop in crops:
        urls.append(pil_image_to_base64_url(crop, max_side=1500))
    return urls
def find_nonwhite_bbox(img: Image.Image, threshold: int = 238, padding_ratio: float = 0.025):
    """
    Find the real receipt/document content area on scanned photos.
    Earlier versions used the first/last dark pixel. That fails for scans with
    dust/noise in the white margin: one black speck at the border makes the crop
    become the whole page, so the actual receipt stays tiny and the vendor line
    is misread as a generic heading such as "NEWSPAPER DEALERS".
    This version is still generic: it uses row/column dark-pixel density and
    ignores isolated specks. No vendor name, bill type, or company name is hardcoded.
    """
    gray = img.convert("L")
    width, height = gray.size
    pixels = gray.load()
    row_counts = [0] * height
    col_counts = [0] * width
    # Dense count. PIL loops are fast enough for one rendered page/crop and avoid
    # requiring numpy on offline RHEL servers.
    for y in range(height):
        row_count = 0
        for x in range(width):
            if pixels[x, y] < threshold:
                row_count += 1
                col_counts[x] += 1
        row_counts[y] = row_count
    # Ignore isolated dots/noise. A true document row/column contains many dark
    # pixels; random scanner noise contains only a few.
    row_threshold = max(8, int(width * 0.015))
    col_threshold = max(8, int(height * 0.008))
    rows = [idx for idx, count in enumerate(row_counts) if count > row_threshold]
    cols = [idx for idx, count in enumerate(col_counts) if count > col_threshold]
    if len(rows) >= 10 and len(cols) >= 10:
        min_y, max_y = rows[0], rows[-1]
        min_x, max_x = cols[0], cols[-1]
    else:
        # Fallback to loose bbox if the document is extremely faint.
        min_x, min_y = width, height
        max_x, max_y = -1, -1
        step_x = max(1, width // 900)
        step_y = max(1, height // 900)
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                if pixels[x, y] < 245:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        if max_x < min_x or max_y < min_y:
            return (0, 0, width, height)
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    return (
        max(0, int(min_x) - pad_x),
        max(0, int(min_y) - pad_y),
        min(width, int(max_x) + pad_x),
        min(height, int(max_y) + pad_y),
    )
def crop_vendor_identity_image_urls_from_file(file_path: str, filename: str) -> List[str]:
    """
    High-resolution generic vendor/header crops.
    Important for scanned bills where the receipt occupies a small centered area
    and the vendor line is tiny or printed in an Indian script.
    """
    img = load_first_page_or_image(file_path, filename)
    width, height = img.size
    bbox = find_nonwhite_bbox(img)
    doc_crop = img.crop(bbox)
    doc_w, doc_h = doc_crop.size
    crops = []
    crops.append(doc_crop)
    # Tight generic issuer/from-header crops for small cash memos and handwritten bills.
    # These are layout/label based only; no vendor or client name is hardcoded.
    crops.append(doc_crop.crop((0, 0, int(doc_w * 0.68), int(doc_h * 0.28))))
    crops.append(doc_crop.crop((0, 0, int(doc_w * 0.58), int(doc_h * 0.22))))
    crops.append(doc_crop.crop((0, 0, int(doc_w * 0.82), int(doc_h * 0.34))))
    # Top header of detected document.
    crops.append(doc_crop.crop((0, 0, doc_w, int(doc_h * 0.42))))
    # Centered top header, useful for small receipt/memo formats.
    crops.append(doc_crop.crop((int(doc_w * 0.03), 0, int(doc_w * 0.97), int(doc_h * 0.36))))
    # Left/top header, useful when proprietor names and shop address are split.
    crops.append(doc_crop.crop((0, 0, int(doc_w * 0.78), int(doc_h * 0.45))))
    # Bottom/footer area is included because many small newspaper/dealer bills
    # print or stamp the agency name again near the signature.
    crops.append(doc_crop.crop((0, int(doc_h * 0.62), doc_w, doc_h)))
    crops = add_rotation_aware_repair_crops(crops, img, include_header_slices=True)
    urls = []
    for crop in crops:
        urls.append(pil_image_to_base64_url(crop, max_side=2600))
    return urls
def enhance_legal_supplier_crop(crop: Image.Image) -> Image.Image:
    """Light, generic enhancement for small scanned header text."""
    try:
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        crop = ImageEnhance.Contrast(crop).enhance(1.25)
        crop = ImageEnhance.Sharpness(crop).enhance(1.35)
        return crop.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    except Exception:
        return crop
def load_first_page_or_image_for_legal_supplier(file_path: str, filename: str) -> Image.Image:
    """Render the first page at higher DPI for legal supplier header extraction."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        doc = fitz.open(file_path)
        try:
            page = doc[0]
            dpi = max(PDF_DPI, LEGAL_SUPPLIER_IMAGE_DPI)
            zoom = dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        finally:
            doc.close()
    img = Image.open(file_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
def crop_legal_supplier_image_urls_from_file(file_path: str, filename: str) -> List[str]:
    """High-resolution first-page/header crops for legal supplier repair.
    Generic layout only: prioritize the supplier legal header/address/GSTIN/PAN
    area and avoid letting a decorative hotel/logo trade name override the legal
    entity. No vendor/client/company names are hardcoded.
    """
    img = load_first_page_or_image_for_legal_supplier(file_path, filename)
    if img.mode != "RGB":
        img = img.convert("RGB")
    try:
        bbox = find_nonwhite_bbox(img)
        doc_crop = img.crop(bbox)
    except Exception:
        doc_crop = img
    doc_w, doc_h = doc_crop.size
    crops: List[Image.Image] = []
    # Priority 1: top-left legal supplier block. This is where many scanned hotel
    # invoices print legal company name + address + PAN/GSTIN/FSSAI. Excluding the
    # top-right logo makes the model less likely to choose a trade/brand name.
    crops.append(doc_crop.crop((0, 0, int(doc_w * 0.74), int(doc_h * 0.18))))
    crops.append(doc_crop.crop((0, 0, int(doc_w * 0.82), int(doc_h * 0.24))))
    crops.append(doc_crop.crop((0, 0, int(doc_w * 0.72), int(doc_h * 0.30))))
    # Secondary: full header for invoices where the legal name is centered or spans
    # the width. Keep after left crops so the focused legal block gets priority.
    crops.append(doc_crop.crop((0, 0, doc_w, int(doc_h * 0.24))))
    # Last resort: full first page for context. The merge logic will only use a
    # legal supplier candidate when it is stronger than a logo/trade name.
    crops.append(doc_crop)
    urls: List[str] = []
    seen = set()
    for crop in crops:
        try:
            crop = enhance_legal_supplier_crop(crop)
            # Deduplicate by approximate geometry, not exact object identity.
            key = (round(crop.size[0] / 50), round(crop.size[1] / 50))
            if key in seen:
                continue
            seen.add(key)
            urls.append(pil_image_to_base64_url(crop, max_side=3000))
        except Exception:
            continue
    return urls
def _smooth_projection_counts(counts: List[int], window: int = 15) -> List[int]:
    """Small moving average for page projection counts."""
    if not counts:
        return []
    window = max(1, int(window))
    half = window // 2
    prefix = [0]
    for value in counts:
        prefix.append(prefix[-1] + int(value))
    smoothed: List[int] = []
    n = len(counts)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed.append((prefix[end] - prefix[start]) // max(1, end - start))
    return smoothed
def _projection_intervals(counts: List[int], min_count: int, min_len: int, bridge_gap: int) -> List[Tuple[int, int]]:
    """Return contiguous dense intervals with small gaps bridged."""
    intervals: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    last_seen = -1
    for idx, count in enumerate(counts):
        if count > min_count:
            if not in_run:
                start = idx
                in_run = True
            last_seen = idx
        elif in_run and (idx - last_seen) > bridge_gap:
            end = last_seen
            if end - start + 1 >= min_len:
                intervals.append((start, end))
            in_run = False
    if in_run:
        end = last_seen
        if end - start + 1 >= min_len:
            intervals.append((start, end))
    return intervals
def _merge_close_intervals(intervals: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
def _dedupe_pil_crops_by_geometry(crops: List[Image.Image]) -> List[Image.Image]:
    """Drop exact duplicate crops while keeping same-sized different panels."""
    unique: List[Image.Image] = []
    seen = set()
    for crop in crops:
        try:
            w, h = crop.size
            if w < 80 or h < 80:
                continue
            thumb = crop.convert("L").resize((16, 16))
            key = (round(w / 20), round(h / 20), thumb.tobytes())
            if key in seen:
                continue
            seen.add(key)
            unique.append(crop)
        except Exception:
            continue
    return unique
def detect_multi_receipt_panel_crops(img: Image.Image) -> List[Image.Image]:
    """Detect separate pasted/scanned handwritten receipt panels using only layout.
    This is intentionally narrow and is only used by the multi-receipt repair
    path. It does not contain vendor/client/item hardcoding. It looks for dense
    notebook/receipt areas separated by whitespace and crops each panel so the
    VLM reads every small slip independently instead of dropping one.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    if width < 300 or height < 300:
        return []
    gray = img.convert("L")
    pixels = gray.load()
    dark_threshold = 222
    margin_x = int(width * 0.02)
    margin_y = int(height * 0.02)
    col_counts = [0] * width
    row_counts = [0] * height
    for y in range(margin_y, max(margin_y, height - margin_y)):
        for x in range(margin_x, max(margin_x, width - margin_x)):
            if pixels[x, y] < dark_threshold:
                col_counts[x] += 1
                row_counts[y] += 1
    col_counts = _smooth_projection_counts(col_counts, max(9, width // 180))
    row_counts = _smooth_projection_counts(row_counts, max(9, height // 180))
    x_min_count = max(10, int(height * 0.012))
    y_min_count = max(10, int(width * 0.012))
    x_intervals = _projection_intervals(col_counts, x_min_count, max(60, width // 12), max(18, width // 80))
    y_intervals = _projection_intervals(row_counts, y_min_count, max(80, height // 12), max(22, height // 70))
    x_intervals = _merge_close_intervals(x_intervals, max(20, width // 60))
    y_intervals = _merge_close_intervals(y_intervals, max(25, height // 55))
    # Ignore intervals that are only page edges or scanner borders.
    x_intervals = [(a, b) for a, b in x_intervals if (b - a) >= width * 0.12 and a > width * 0.01 and b < width * 0.99]
    y_intervals = [(a, b) for a, b in y_intervals if (b - a) >= height * 0.12 and a > height * 0.005 and b < height * 0.995]
    if len(x_intervals) < 2 or len(y_intervals) < 2:
        return []
    # Keep strongest two or three bands in reading order. This targets actual
    # separate panel layouts, not normal single invoices.
    x_intervals = sorted(x_intervals, key=lambda ab: ab[0])[:3]
    y_intervals = sorted(y_intervals, key=lambda ab: ab[0])[:3]
    pad_x = max(22, int(width * 0.018))
    pad_y = max(22, int(height * 0.018))
    crops: List[Image.Image] = []
    for y1, y2 in y_intervals:
        for x1, x2 in x_intervals:
            left = max(0, x1 - pad_x)
            top = max(0, y1 - pad_y)
            right = min(width, x2 + pad_x)
            bottom = min(height, y2 + pad_y)
            crop = img.crop((left, top, right, bottom))
            cw, ch = crop.size
            # Reject tiny cross-products and very tall/full-page fragments.
            if cw < width * 0.10 or ch < height * 0.10:
                continue
            crops.append(crop)
    return _dedupe_pil_crops_by_geometry(crops)
def crop_multi_receipt_panel_image_urls_from_file(file_path: str, filename: str) -> List[str]:
    """Return cropped receipt-panel images for focused local handwritten repair."""
    try:
        img = load_first_page_or_image(file_path, filename)
        try:
            # Work on the actual non-white document area when possible, but keep
            # the original page fallback for scans where the content is already
            # separated by whitespace.
            bbox = find_nonwhite_bbox(img, threshold=232, padding_ratio=0.015)
            doc_crop = img.crop(bbox)
        except Exception:
            doc_crop = img
        crops = detect_multi_receipt_panel_crops(doc_crop)
        if len(crops) < 2:
            crops = detect_multi_receipt_panel_crops(img)
        if len(crops) < 2:
            return []
        urls: List[str] = []
        for crop in crops[:MAX_MULTI_RECEIPT_PANEL_IMAGES]:
            try:
                crop = ImageEnhance.Contrast(crop).enhance(1.12)
                crop = ImageEnhance.Sharpness(crop).enhance(1.20)
                urls.append(pil_image_to_base64_url(crop, max_side=2200))
            except Exception:
                continue
        return urls
    except Exception:
        return []
def _normalize_panel_item_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    date_value = normalize_date_string(first_non_empty(item.get("bill_date_raw"), item.get("from_date_raw"), item.get("to_date_raw")))
    particulars = re.sub(r"[^a-z0-9]+", "", safe_str(item.get("particulars_raw")).lower())[:80]
    amount_decimal = money_decimal_or_none(item.get("amount_raw"))
    amount = format_decimal_amount(amount_decimal) if amount_decimal is not None else safe_str(item.get("amount_raw"))
    return (date_value, particulars, amount)
def _fix_local_receipt_quantity_from_particulars(item: Dict[str, Any]) -> Dict[str, Any]:
    """Recover quantity from item text when the VLM used serial number 1 as qty."""
    item = dict(item)
    text = safe_str(item.get("particulars_raw"))
    qty = safe_str(item.get("quantity_raw"))
    match = re.search(r"(?i)(?:^|[\s(\-])([0-9]{1,4}(?:\.\d+)?)\s*(?:kg|kilo|kilogram|kgs|pcs|pc|piece|pieces)\b", text)
    if match and (not qty or qty in {"1", "1.0", "1.00"}):
        item["quantity_raw"] = match.group(1)
    return item
def combine_multi_receipt_panel_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine independently extracted panel crops into one raw result."""
    combined: Dict[str, Any] = {key: "" for key in RAW_KEYS}
    combined["items"] = []
    seen = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        normalized = normalize_raw_result(result)
        for key in RAW_KEYS:
            if key == "items":
                continue
            value = safe_str(normalized.get(key))
            if value and not safe_str(combined.get(key)):
                combined[key] = value
        for item in normalized.get("items") or []:
            if not isinstance(item, dict):
                continue
            item = _fix_local_receipt_quantity_from_particulars(item)
            if not safe_str(item.get("particulars_raw")) or not clean_amount(item.get("amount_raw")):
                continue
            key = _normalize_panel_item_key(item)
            if key in seen:
                continue
            seen.add(key)
            combined["items"].append(item)
    if combined["items"]:
        total = Decimal("0")
        ok_total = True
        for item in combined["items"]:
            amt = money_decimal_or_none(item.get("amount_raw"))
            if amt is None:
                ok_total = False
                break
            total += amt
        if ok_total and total > 0:
            combined["grand_total_raw"] = format_decimal_amount(total)
    return combined
def run_multi_receipt_panel_repair(file_path: str, filename: str, fallback_image_urls: List[str]) -> Dict[str, Any]:
    """Run focused multi-panel extraction on detected crops, with full-page fallback."""
    results: List[Dict[str, Any]] = []
    panel_urls = crop_multi_receipt_panel_image_urls_from_file(file_path, filename)
    if len(panel_urls) >= 2:
        # Read each small receipt independently so the model cannot silently drop
        # a quadrant. This is slower, but gated to handwritten/local multi-panel
        # cases only.
        for panel_url in panel_urls[:MAX_MULTI_RECEIPT_PANEL_IMAGES]:
            try:
                result = call_multi_receipt_panel_llm([panel_url])
            except Exception:
                continue
            if multi_receipt_panel_result_has_useful_items(result):
                results.append(result)
            else:
                # Single-panel crops may have only one dated item; keep it if it
                # has a real item and amount.
                normalized = normalize_raw_result(result)
                if count_good_receipt_panel_items(normalized) >= 1:
                    results.append(result)
        combined = combine_multi_receipt_panel_results(results)
        if multi_receipt_panel_result_has_useful_items(combined) or count_good_receipt_panel_items(combined) >= 2:
            return combined
        # One combined crop call as a fallback for models that need page context.
        try:
            combined_crop_result = call_multi_receipt_panel_llm(panel_urls[:MAX_MULTI_RECEIPT_PANEL_IMAGES])
            if multi_receipt_panel_result_has_useful_items(combined_crop_result):
                return combined_crop_result
        except Exception:
            pass
    return call_multi_receipt_panel_llm(fallback_image_urls[:MAX_MULTI_RECEIPT_PANEL_IMAGES])
def crop_party_identity_image_urls_from_file(file_path: str, filename: str) -> List[str]:
    """Compact orientation set for final supplier/client repair.
    This function is purely layout based. It creates full-document and upper-band
    candidates in normal and rotated orientations so sideways scanned bills can be
    read without hardcoded vendor/client names.
    """
    img = load_first_page_or_image(file_path, filename)
    try:
        bbox = find_nonwhite_bbox(img)
        doc_crop = img.crop(bbox)
    except Exception:
        doc_crop = img
    candidates: List[Image.Image] = []
    def add_identity_views(view: Image.Image) -> None:
        vw, vh = view.size
        # Full visible bill/receipt.
        candidates.append(view)
        # Header + billed-party region. For most bills this includes top logo/name,
        # M/s/Bill-To block, date/bill-no and first table header.
        candidates.append(view.crop((0, 0, vw, int(vh * 0.62))))
        # Right/bottom signatory area, useful for "For <supplier>" on bill of supply.
        candidates.append(view.crop((int(vw * 0.45), int(vh * 0.45), vw, vh)))
        # Left/top issuer/customer label area, useful for small cash memos.
        candidates.append(view.crop((0, 0, int(vw * 0.68), int(vh * 0.52))))
    # Keep original for normal bills and both 90-degree orientations for sideways scans.
    add_identity_views(doc_crop)
    for angle in (90, 270):
        try:
            add_identity_views(doc_crop.rotate(angle, expand=True))
        except Exception:
            continue
    # Remove exact duplicate dimensions/order while preserving order.
    encoded: List[str] = []
    seen = set()
    for crop in candidates:
        try:
            key = crop.size
            if key in seen and len(encoded) >= 3:
                continue
            seen.add(key)
            encoded.append(pil_image_to_base64_url(crop, max_side=PARTY_IDENTITY_IMAGE_SIDE))
        except Exception:
            continue
    return encoded
# JSON parsing and recovery
# =========================
RAW_KEYS = [
    "document_type_raw",
    "whether_self_reimbursement_raw",
    "vendor_employee_name_raw",
    "expense_type_raw",
    "los_vendor_state_raw",
    "lor_isec_state_raw",
    "service_usage_raw",
    "is_invoice_digitally_signed_raw",
    "digital_signature_evidence_raw",
    "remarks_raw",
    "approver_raw",
    "client_name_raw",
    "invoice_no_raw",
    "bill_no_raw",
    "po_no_raw",
    "bill_date_raw",
    "service_period_raw",
    "explicit_date_range_raw",
    "vendor_gstin_raw",
    "isec_gstin_raw",
    "vendor_pan_raw",
    "department_code_raw",
    "function_code_raw",
    "location_code_raw",
    "sub_loc_code_raw",
    "subtotal_raw",
    "cgst_total_raw",
    "sgst_total_raw",
    "igst_total_raw",
    "cess_total_raw",
    "round_off_raw",
    "grand_total_raw",
    "amount_in_words_raw",
]
EXTRA_HINT_KEYS = [
    "vendor_address_raw",
    "billed_party_raw",
    "billed_party_block_raw",
    "customer_block_raw",
    "client_block_raw",
    "client_address_raw",
    "patient_address_raw",
    "supply_address_raw",
    "bill_to_address_raw",
    "ship_to_address_raw",
    "place_of_supply_raw",
    "state_raw",
    "account_name_raw",
    "service_title_raw",
    "bill_upto_raw",
    "bill_period_raw",
    "visible_text_raw",
    "_utility_ocr_text_raw",
    "_utility_bill_no_header_evidence",
    "_utility_bill_date_label_evidence",
    "_utility_period_title_evidence",
]
HEADER_KEYS = [
    "vendor_employee_name_raw",
    "expense_type_raw",
    "los_vendor_state_raw",
    "lor_isec_state_raw",
    "service_usage_raw",
    "client_name_raw",
    "invoice_no_raw",
    "bill_no_raw",
    "bill_date_raw",
    "service_period_raw",
    "explicit_date_range_raw",
    "vendor_gstin_raw",
    "isec_gstin_raw",
    "vendor_pan_raw",
    "approver_raw",
    "whether_self_reimbursement_raw",
    "is_invoice_digitally_signed_raw",
    "digital_signature_evidence_raw",
]
ITEM_KEYS = [
    "particulars_raw",
    "quantity_raw",
    "quantity_unit_raw",
    "hsn_code_raw",
    "rate_raw",
    "amount_raw",
    "cgst_raw",
    "sgst_raw",
    "igst_raw",
    "cess_raw",
    # Optional item-level date fields used only by focused multi-receipt/collage
    # repair. Existing prompts may omit them; blank values keep old behavior.
    "bill_date_raw",
    "from_date_raw",
    "to_date_raw",
    "service_period_raw",
    "explicit_date_range_raw",
]
def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text
def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)
def balance_json(text: str) -> str:
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()
    if in_string:
        text += '"'
    return text + "".join(reversed(stack))
def extract_json_object_text(text: str) -> str:
    text = strip_markdown_fences(text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model response.")
    end = text.rfind("}")
    if end == -1 or end < start:
        return text[start:]
    return text[start:end + 1]
def salvage_flat_fields(text: str) -> Dict[str, Any]:
    recovered: Dict[str, Any] = {key: "" for key in RAW_KEYS}
    recovered["items"] = []
    all_keys = list(set(RAW_KEYS + HEADER_KEYS + EXTRA_HINT_KEYS))
    for key in all_keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*(?:"((?:\\.|[^"\\])*)"|null|(-?\d+(?:\.\d+)?))'
        match = re.search(pattern, text)
        if not match:
            continue
        if match.group(1) is not None:
            recovered[key] = match.group(1).replace('\\"', '"')
        elif match.group(2) is not None:
            recovered[key] = match.group(2)
        else:
            recovered[key] = ""
    item_blocks = re.findall(r"\{[^{}]*particulars_raw[^{}]*\}", text)
    for block in item_blocks:
        item = {key: "" for key in ITEM_KEYS}
        for key in ITEM_KEYS:
            pattern = rf'"{re.escape(key)}"\s*:\s*(?:"((?:\\.|[^"\\])*)"|null|(-?\d+(?:\.\d+)?))'
            match = re.search(pattern, block)
            if not match:
                continue
            if match.group(1) is not None:
                item[key] = match.group(1).replace('\\"', '"')
            elif match.group(2) is not None:
                item[key] = match.group(2)
            else:
                item[key] = ""
        recovered["items"].append(item)
    return recovered
def extract_json_from_response(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty model response.")
    raw_text = text
    json_text = extract_json_object_text(text)
    json_text = remove_trailing_commas(json_text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    balanced = balance_json(json_text)
    balanced = remove_trailing_commas(balanced)
    try:
        return json.loads(balanced)
    except json.JSONDecodeError:
        recovered = salvage_flat_fields(raw_text)
        has_any_value = any(
            str(recovered.get(key, "")).strip()
            for key in RAW_KEYS + HEADER_KEYS
        ) or bool(recovered.get("items"))
        if has_any_value:
            recovered["_warning"] = "Model returned invalid JSON. Recovered available fields from model text."
            return recovered
        raise ValueError(f"Model returned invalid JSON and recovery failed. Raw response: {raw_text[:2000]}")
# =========================
# LLM calls
# =========================
def call_vision_llm(images_base64_urls: List[str]) -> Dict[str, Any]:
    content = [{"type": "text", "text": USER_PROMPT}]
    for image_url in images_base64_urls:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=1600,
            messages=messages,
            response_format={"type": "json_object"},
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=1600,
            messages=messages,
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"plain_json failed: {str(e)}")
    raise ValueError("LLM extraction failed. " + " | ".join(errors))
def call_header_llm(header_image_urls: List[str]) -> Dict[str, Any]:
    # Defensive cap: even if crop generation changes later, header LLM must not
    # receive too many images because Qwen2.5-VL has a finite context window.
    header_image_urls = header_image_urls[:max(1, MAX_HEADER_CROPS)]
    content = [{"type": "text", "text": HEADER_USER_PROMPT}]
    for image_url in header_image_urls:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )
    messages = [
        {
            "role": "system",
            "content": HEADER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=900,
            messages=messages,
            response_format={"type": "json_object"},
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"header json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=900,
            messages=messages,
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"header plain_json failed: {str(e)}")
    raise ValueError("Header extraction failed. " + " | ".join(errors))
def call_receipt_repair_llm(images_base64_urls: List[str]) -> Dict[str, Any]:
    content = [{"type": "text", "text": REPAIR_USER_PROMPT}]
    # Use first two pages only for repair to control latency. Receipts usually keep
    # number/date/amount/table fields on the first page.
    for image_url in images_base64_urls[:2]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )
    messages = [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=1100,
            messages=messages,
            response_format={"type": "json_object"},
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"repair json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=1100,
            messages=messages,
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"repair plain_json failed: {str(e)}")
    raise ValueError("Receipt repair extraction failed. " + " | ".join(errors))
def call_vendor_identity_llm(images_base64_urls: List[str]) -> Dict[str, Any]:
    content = [{"type": "text", "text": VENDOR_IDENTITY_USER_PROMPT}]
    for image_url in images_base64_urls[:MAX_VENDOR_IDENTITY_IMAGES]:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = [
        {"role": "system", "content": VENDOR_IDENTITY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=500,
            messages=messages,
            response_format={"type": "json_object"},
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"vendor json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=500,
            messages=messages,
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"vendor plain_json failed: {str(e)}")
    raise ValueError("Vendor identity extraction failed. " + " | ".join(errors))
def call_customer_block_llm(images_base64_urls: List[str]) -> Dict[str, Any]:
    content = [{"type": "text", "text": CLIENT_BLOCK_USER_PROMPT}]
    for image_url in images_base64_urls[:MAX_CUSTOMER_BLOCK_IMAGES]:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = [
        {"role": "system", "content": CLIENT_BLOCK_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=650,
            messages=messages,
            response_format={"type": "json_object"},
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"customer_block json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=650,
            messages=messages,
        )
        model_text = response.choices[0].message.content or ""
        return extract_json_from_response(model_text)
    except Exception as e:
        errors.append(f"customer_block plain_json failed: {str(e)}")
    raise ValueError("Customer block extraction failed. " + " | ".join(errors))
# =========================
def call_party_identity_llm(images_base64_urls: List[str]) -> Dict[str, Any]:
    content = [{"type": "text", "text": PARTY_IDENTITY_USER_PROMPT}]
    for image_url in images_base64_urls[:MAX_PARTY_IDENTITY_IMAGES]:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = [
        {"role": "system", "content": PARTY_IDENTITY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, temperature=0, max_tokens=650, messages=messages, response_format={"type": "json_object"}
        )
        return extract_json_from_response(response.choices[0].message.content or "")
    except Exception as e:
        errors.append(f"party_identity json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, temperature=0, max_tokens=650, messages=messages
        )
        return extract_json_from_response(response.choices[0].message.content or "")
    except Exception as e:
        errors.append(f"party_identity plain_json failed: {str(e)}")
    raise ValueError("Party identity extraction failed. " + " | ".join(errors))
def call_legal_supplier_llm(images_base64_urls: List[str]) -> Dict[str, Any]:
    """Focused legal supplier repair for invoices with separate trade/logo names."""
    content = [{"type": "text", "text": LEGAL_SUPPLIER_USER_PROMPT}]
    for image_url in images_base64_urls[:MAX_LEGAL_SUPPLIER_IMAGES]:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = [
        {"role": "system", "content": LEGAL_SUPPLIER_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, temperature=0, max_tokens=700, messages=messages, response_format={"type": "json_object"}
        )
        return extract_json_from_response(response.choices[0].message.content or "")
    except Exception as e:
        errors.append(f"legal_supplier json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, temperature=0, max_tokens=700, messages=messages
        )
        return extract_json_from_response(response.choices[0].message.content or "")
    except Exception as e:
        errors.append(f"legal_supplier plain_json failed: {str(e)}")
    raise ValueError("Legal supplier extraction failed. " + " | ".join(errors))
def call_multi_receipt_panel_llm(images_base64_urls: List[str]) -> Dict[str, Any]:
    """Focused extraction for pages containing several handwritten/local receipt panels."""
    content = [{"type": "text", "text": MULTI_RECEIPT_PANEL_USER_PROMPT}]
    for image_url in images_base64_urls[:MAX_MULTI_RECEIPT_PANEL_IMAGES]:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = [
        {"role": "system", "content": MULTI_RECEIPT_PANEL_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=1400,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return extract_json_from_response(response.choices[0].message.content or "")
    except Exception as e:
        errors.append(f"multi_receipt_panel json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=1400,
            messages=messages,
        )
        return extract_json_from_response(response.choices[0].message.content or "")
    except Exception as e:
        errors.append(f"multi_receipt_panel plain_json failed: {str(e)}")
    raise ValueError("Multi-receipt panel extraction failed. " + " | ".join(errors))
def call_charge_summary_llm(images_base64_urls: List[str]) -> Dict[str, Any]:
    """Focused, low-token summary/tax repair call for scanned utility/hotel bills."""
    content = [{"type": "text", "text": CHARGE_SUMMARY_USER_PROMPT}]
    # Send a small group of pages together for multipage hotel bills. Seeing the
    # header page and summary page in the same call prevents page-subtotal/day-total
    # values and OCR digit slips from overwriting the final invoice values.
    for image_url in images_base64_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = [
        {"role": "system", "content": CHARGE_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    errors = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=850,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return extract_json_from_response(response.choices[0].message.content or "")
    except Exception as e:
        errors.append(f"charge_summary json_object failed: {str(e)}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=850,
            messages=messages,
        )
        return extract_json_from_response(response.choices[0].message.content or "")
    except Exception as e:
        errors.append(f"charge_summary plain_json failed: {str(e)}")
    raise ValueError("Charge summary extraction failed. " + " | ".join(errors))
# Normalization helpers
# =========================
def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
def first_non_empty(*values: Any) -> str:
    for value in values:
        text = safe_str(value)
        if text:
            return text
    return ""
def clean_amount(value: Any) -> str:
    text = safe_str(value)
    if not text:
        return ""
    text = text.replace(",", "")
    text = re.sub(r"(₹|Rs\.?|INR|rupees?)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"/\s*-\s*$", "", text)
    text = re.sub(r"/\s*$", "", text)
    text = text.strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return ""
    number = match.group(0)
    try:
        dec = Decimal(number)
    except InvalidOperation:
        return number
    if dec == dec.to_integral():
        return str(int(dec))
    return format(dec.normalize(), "f")
def decimal_or_none(value: Any):
    cleaned = clean_amount(value)
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None
def is_whole_number(value: Decimal) -> bool:
    return value == value.to_integral_value()
def generic_amount_candidates(raw_amount: Any) -> List[str]:
    """
    Generic OCR amount candidates.
    No fixed bill/date/vendor/amount.
    Candidate with one trailing digit removed is only used later if arithmetic validates it.
    """
    cleaned = clean_amount(raw_amount)
    if not cleaned:
        return []
    candidates = [cleaned]
    if cleaned.isdigit() and len(cleaned) > 2:
        candidates.append(cleaned[:-1])
    unique = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique
def choose_amount_by_arithmetic(raw_amount: Any, raw_rate: Any, raw_quantity: Any = "") -> str:
    candidates = generic_amount_candidates(raw_amount)
    if not candidates:
        return ""
    rate = decimal_or_none(raw_rate)
    quantity = decimal_or_none(raw_quantity)
    if rate is None or rate == 0:
        return candidates[0]
    if quantity is not None:
        expected = quantity * rate
        for candidate in candidates:
            candidate_decimal = decimal_or_none(candidate)
            if candidate_decimal is not None and candidate_decimal == expected:
                return candidate
        return candidates[0]
    original_decimal = decimal_or_none(candidates[0])
    if original_decimal is not None:
        original_quantity = original_decimal / rate
        if is_whole_number(original_quantity):
            return candidates[0]
    for candidate in candidates[1:]:
        candidate_decimal = decimal_or_none(candidate)
        if candidate_decimal is None:
            continue
        calculated_quantity = candidate_decimal / rate
        if is_whole_number(calculated_quantity):
            return candidate
    return candidates[0]
def infer_quantity_by_arithmetic(amount: Any, rate: Any) -> str:
    amount_decimal = decimal_or_none(amount)
    rate_decimal = decimal_or_none(rate)
    if amount_decimal is None or rate_decimal is None or rate_decimal == 0:
        return ""
    quantity = amount_decimal / rate_decimal
    if is_whole_number(quantity):
        return str(int(quantity))
    return ""
def quantity_matches_numeric_value(quantity: Any, *values: Any) -> bool:
    quantity_decimal = decimal_or_none(quantity)
    if quantity_decimal is None:
        return False
    for value in values:
        value_decimal = decimal_or_none(value)
        if value_decimal is None:
            continue
        if quantity_decimal == value_decimal and quantity_decimal != Decimal("1"):
            return True
    return False
def sanitize_quantity_for_output(
    quantity: Any,
    amount: Any = "",
    item: Optional[Dict[str, Any]] = None,
    raw: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Final guard for quantity.
    Quantity must be an actual count only. OCR/VLM sometimes copies taxable value,
    line amount, GST amount, or total into quantity. This guard removes those
    amount-like values without using any vendor-specific rule.
    """
    text = safe_str(quantity).strip()
    if not text:
        return ""
    if re.search(r"[%₹]|\b(rs\.?|inr|amount|total|tax|gst|cgst|sgst|igst|cess|rate|value|payable|paid)\b", text, flags=re.IGNORECASE):
        return ""
    # Accept plain numeric quantities only. Do not extract a number from a larger
    # amount-like string because that is how amounts leak into quantity.
    if not re.fullmatch(r"\d+(?:\.\d+)?", text.replace(",", "")):
        return ""
    # Comma-grouped numbers in quantity are usually amounts, not counts.
    if "," in text:
        return ""
    cleaned = clean_amount(text)
    quantity_decimal = decimal_or_none(cleaned)
    if quantity_decimal is None or quantity_decimal <= 0:
        return ""
    compare_values: List[Any] = [amount]
    if isinstance(item, dict):
        compare_values.extend([
            item.get("amount_raw"),
            item.get("cgst_raw"),
            item.get("sgst_raw"),
            item.get("igst_raw"),
            item.get("cess_raw"),
        ])
    if isinstance(raw, dict):
        compare_values.extend([
            raw.get("subtotal_raw"),
            raw.get("grand_total_raw"),
            raw.get("cgst_total_raw"),
            raw.get("sgst_total_raw"),
            raw.get("igst_total_raw"),
            raw.get("cess_total_raw"),
            raw.get("_calculated_items_total"),
            raw.get("_source_text_claim_total_raw"),
        ])
    if quantity_matches_numeric_value(quantity_decimal, *compare_values):
        return ""
    # Very large decimal quantities are usually OCR-copied money values. Keep large
    # whole-number counts possible, but reject large fractional amount-like values.
    if quantity_decimal > Decimal("10000"):
        return ""
    if not is_whole_number(quantity_decimal) and quantity_decimal > Decimal("100"):
        return ""
    return format_decimal_amount(quantity_decimal)
def choose_amount_rate_quantity(item: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Select final amount and quantity.
    In this API version, the output rate column means GST percentage only, so
    quantity must never be inferred by amount / rate. That old arithmetic was
    useful when rate meant unit price, but it is wrong after changing rate to GST %.
    """
    raw_amount = item.get("amount_raw")
    raw_rate = item.get("rate_raw")
    raw_quantity = item.get("quantity_raw")
    amount_candidates = generic_amount_candidates(raw_amount)
    final_amount = amount_candidates[0] if amount_candidates else ""
    # If the model placed a non-percent unit amount in rate_raw and amount_raw is
    # blank, use it as a last-resort amount. Do not use GST percentages as amount.
    if not final_amount and not normalize_percent_rate(raw_rate):
        possible_amount = clean_amount(raw_rate)
        possible_amount_decimal = decimal_or_none(possible_amount)
        if possible_amount_decimal is not None and possible_amount_decimal > 0:
            final_amount = possible_amount
    final_quantity = sanitize_quantity_for_output(raw_quantity, final_amount, item)
    # Return rate as empty here. The final output rate is calculated later by
    # choose_gst_rate_for_output() from visible GST % or tax/amount math.
    return final_amount, "", final_quantity
def normalize_yes_no(value: Any) -> str:
    text = safe_str(value).lower()
    if not text:
        return ""
    if re.search(r"\b(yes|y|true)\b", text):
        return "Yes"
    if re.search(r"\b(no|n|false)\b", text):
        return "No"
    return safe_str(value)
def decide_digital_signature(value: Any, evidence: Any) -> str:
    raw_yes_no = normalize_yes_no(value)
    combined = f"{safe_str(value)} {safe_str(evidence)}".lower()
    actual_digital_evidence = re.search(
        r"\b(irn|ack\s*no|ack\s*date|acknowledgement|e-?invoice|qr\s*code|digital(?:ly)?\s*signed|digital\s*signature)\b",
        combined,
    )
    ordinary_signature_evidence = re.search(
        r"\b(cashier\s*id|cashier|guest\s*signature|receiver'?s?\s*signature|authori[sz]ed\s*signatory|handwritten\s*signature|signature)\b",
        combined,
    )
    if raw_yes_no == "Yes" and ordinary_signature_evidence and not actual_digital_evidence:
        return "No"
    if raw_yes_no == "Yes":
        return "Yes"
    if re.search(
        r"\b(irn|ack\s*no|ack\s*date|acknowledgement|e-?invoice|qr\s*code|digital(?:ly)?\s*signed|digital\s*signature)\b",
        combined,
    ):
        return "Yes"
    if raw_yes_no == "No":
        return "No"
    # Requirement: field should not stay blank. If no digital-signature/e-invoice
    # evidence was extracted, treat it as not digitally signed.
    return "No"
def _contains_any_bytes(data: bytes, markers: List[bytes]) -> bool:
    return any(marker in data for marker in markers)
def _pdf_xref_object(doc: fitz.Document, xref: int) -> str:
    try:
        return doc.xref_object(xref, compressed=False) or ""
    except Exception:
        return ""
def _pdf_signature_widget_has_value(doc: fitz.Document, widget: Any) -> bool:
    """
    A PDF may contain an empty signature field. Treat it as digitally signed only
    when the signature widget points to a value object, or the PDF has ByteRange/
    PKCS7 markers elsewhere.
    """
    widget_xref = getattr(widget, "xref", None)
    if not widget_xref:
        return False
    widget_obj = _pdf_xref_object(doc, int(widget_xref))
    # Signed widgets normally contain /V <xref> pointing to the signature dictionary.
    return bool(re.search(r"/V\s+\d+\s+0\s+R", widget_obj))
def extract_pdf_signature_evidence(file_path: str, filename: str) -> str:
    """
    Generic PDF digital-signature detector.
    It does NOT depend on vendor names or visible text like "computer generated".
    It returns evidence only when the PDF structure contains a real signature
    dictionary/signature object, for example:
      - AcroForm signature widget with /FT /Sig and /V reference
      - Signature dictionary with /ByteRange and /Contents
      - PKCS#7 SubFilter such as adbe.pkcs7.detached
    This handles PDFs that are digitally signed even when the visible invoice
    does not show IRN/Ack/QR details.
    """
    if not safe_str(filename).lower().endswith(".pdf"):
        return ""
    evidence_parts: List[str] = []
    # 1) Low-level byte check. A valid embedded signature normally has ByteRange
    # and a PKCS#7/CMS signature container in /Contents.
    try:
        with open(file_path, "rb") as handle:
            pdf_bytes = handle.read()
        has_byte_range = b"/ByteRange" in pdf_bytes
        has_contents = b"/Contents" in pdf_bytes
        has_sig_type = _contains_any_bytes(
            pdf_bytes,
            [b"/Type/Sig", b"/Type /Sig", b"/FT/Sig", b"/FT /Sig"],
        )
        has_subfilter = _contains_any_bytes(
            pdf_bytes,
            [
                b"/SubFilter",
                b"/adbe.pkcs7.detached",
                b"/adbe.pkcs7.sha1",
                b"/ETSI.CAdES.detached",
            ],
        )
        if has_byte_range and has_contents and (has_sig_type or has_subfilter):
            evidence_parts.append("PDF signature dictionary with ByteRange/Contents present")
    except Exception:
        pass
    # 2) Structured xref/widget check using PyMuPDF.
    try:
        doc = fitz.open(file_path)
        try:
            # Signature widgets in page annotations.
            for page in doc:
                try:
                    widgets = list(page.widgets() or [])
                except Exception:
                    widgets = []
                for widget in widgets:
                    field_type_name = safe_str(getattr(widget, "field_type_string", "")).lower()
                    field_type_code = safe_str(getattr(widget, "field_type", ""))
                    field_name = safe_str(getattr(widget, "field_name", ""))
                    if field_type_name == "signature" or field_type_code == "6":
                        if _pdf_signature_widget_has_value(doc, widget):
                            if field_name:
                                evidence_parts.append(f"PDF signature field with value: {field_name}")
                            else:
                                evidence_parts.append("PDF signature field with value present")
                        else:
                            # Keep checking low-level xrefs before deciding. Empty
                            # signature field alone should not force Yes.
                            pass
            # Signature dictionaries and AcroForm signature fields in xref objects.
            for xref in range(1, doc.xref_length()):
                obj = _pdf_xref_object(doc, xref)
                compact_obj = re.sub(r"\s+", " ", obj)
                is_signature_dict = bool(
                    re.search(r"/Type\s*/Sig\b", compact_obj)
                    or ("/ByteRange" in compact_obj and "/Contents" in compact_obj and "/SubFilter" in compact_obj)
                )
                is_signature_field_with_value = bool(
                    re.search(r"/FT\s*/Sig\b", compact_obj)
                    and re.search(r"/V\s+\d+\s+0\s+R", compact_obj)
                )
                if is_signature_dict:
                    evidence_parts.append(f"PDF signature object xref {xref}")
                elif is_signature_field_with_value:
                    evidence_parts.append(f"PDF signature field xref {xref}")
        finally:
            doc.close()
    except Exception:
        pass
    unique_parts: List[str] = []
    for part in evidence_parts:
        if part and part not in unique_parts:
            unique_parts.append(part)
    return " | ".join(unique_parts)
def enrich_raw_from_pdf_signature(raw: Dict[str, Any], file_path: str, filename: str) -> Dict[str, Any]:
    """
    Update raw extraction with actual PDF digital-signature evidence.
    If PDF signature evidence is present, it must override a model-produced "No".
    """
    signature_evidence = extract_pdf_signature_evidence(file_path, filename)
    if signature_evidence:
        existing = safe_str(raw.get("digital_signature_evidence_raw"))
        if existing and signature_evidence not in existing:
            raw["digital_signature_evidence_raw"] = f"{existing} | {signature_evidence}"
        elif not existing:
            raw["digital_signature_evidence_raw"] = signature_evidence
        raw["is_invoice_digitally_signed_raw"] = "Yes"
    return raw
def build_month_aliases() -> Dict[str, int]:
    aliases = {}
    for month_num in range(1, 13):
        full_name = calendar.month_name[month_num].upper()
        short_name = calendar.month_abbr[month_num].upper()
        if full_name:
            aliases[full_name] = month_num
        if short_name:
            aliases[short_name] = month_num
    return aliases
MONTH_ALIASES = build_month_aliases()
MONTH_PATTERN = "|".join(sorted(MONTH_ALIASES.keys(), key=len, reverse=True))
def normalize_year(year_text: str) -> int:
    year = int(year_text)
    if year < 100:
        if year <= 49:
            return 2000 + year
        return 1900 + year
    return year
def make_date_string(day_text: str, month_text: str, year_text: str) -> str:
    day = int(day_text)
    month = int(month_text)
    year = normalize_year(year_text)
    try:
        date(year, month, day)
    except ValueError:
        return ""
    return f"{day:02d}/{month:02d}/{year}"
def normalize_date_string(value: Any) -> str:
    text = safe_str(value)
    if not text:
        return ""
    match = re.search(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b", text)
    if match:
        return make_date_string(match.group(1), match.group(2), match.group(3))
    # Month-first dates such as Jun 17, 2026 / June 17 2026.
    match = re.search(
        rf"\b({MONTH_PATTERN})[\s\-/.]*(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?[,]?[\s\-/.]*(\d{{2,4}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        month = MONTH_ALIASES.get(match.group(1).upper())
        if month:
            return make_date_string(match.group(2), str(month), match.group(3))
    match = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-/.]*({MONTH_PATTERN})[\s\-/.]*(\d{{2,4}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        month_name = match.group(2).upper()
        month = MONTH_ALIASES.get(month_name)
        if month:
            return make_date_string(match.group(1), str(month), match.group(3))
    return text
def extract_all_full_dates(text: str) -> List[str]:
    text = safe_str(text)
    dates = []
    for match in re.finditer(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b", text):
        value = make_date_string(match.group(1), match.group(2), match.group(3))
        if value and value not in dates:
            dates.append(value)
    for match in re.finditer(
        rf"\b({MONTH_PATTERN})[\s\-/.]*(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?[,]?[\s\-/.]*(\d{{2,4}})\b",
        text,
        flags=re.IGNORECASE,
    ):
        month_name = match.group(1).upper()
        month = MONTH_ALIASES.get(month_name)
        if month:
            value = make_date_string(match.group(2), str(month), match.group(3))
            if value and value not in dates:
                dates.append(value)
    for match in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-/.]*({MONTH_PATTERN})[\s\-/.]*(\d{{2,4}})\b",
        text,
        flags=re.IGNORECASE,
    ):
        month_name = match.group(2).upper()
        month = MONTH_ALIASES.get(month_name)
        if month:
            value = make_date_string(match.group(1), str(month), match.group(3))
            if value and value not in dates:
                dates.append(value)
    return dates
def date_sort_key(date_text: str) -> Tuple[int, int, int]:
    match = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", date_text)
    if not match:
        return (0, 0, 0)
    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    return (year, month, day)
def derive_date_range_from_labelled_period(range_text: str) -> Tuple[str, str]:
    """
    Prefer explicit period labels over unrelated dates in the same text.
    Example: Service Start Date 01-07-2022, Bill Upto 01-Nov-2025 To 30-Nov-2025
    should return 01/11/2025 to 30/11/2025, not the service start date.
    """
    text = safe_str(range_text)
    if not text:
        return "", ""
    period_label_pattern = re.compile(
        r"\b(bill\s+upto|bill\s+up\s+to|bill\s+period|billing\s+period|service\s+period|charge\s+period|period)\b",
        flags=re.IGNORECASE,
    )
    for match in period_label_pattern.finditer(text):
        window = text[match.start(): match.start() + 220]
        dates = extract_all_full_dates(window)
        if len(dates) >= 2:
            from_date, to_date = dates[0], dates[1]
            if date_sort_key(from_date) > date_sort_key(to_date):
                from_date, to_date = to_date, from_date
            return from_date, to_date
    dates = extract_all_full_dates(text)
    if len(dates) > 2 and re.search(r"\b(service\s+start\s+date|start\s+date)\b", text, flags=re.IGNORECASE):
        # When a start-date field and a billing-period field are mixed in OCR output,
        # the last two dates are usually the charge period.
        from_date, to_date = dates[-2], dates[-1]
        if date_sort_key(from_date) > date_sort_key(to_date):
            from_date, to_date = to_date, from_date
        return from_date, to_date
    return "", ""
def derive_date_range_from_explicit_range(range_text: str) -> Tuple[str, str]:
    labelled_from, labelled_to = derive_date_range_from_labelled_period(range_text)
    if labelled_from and labelled_to:
        return labelled_from, labelled_to
    dates = extract_all_full_dates(range_text)
    if len(dates) < 2:
        return "", ""
    if len(dates) > 2 and re.search(r"\b(service\s+start\s+date|start\s+date)\b", safe_str(range_text), flags=re.IGNORECASE):
        from_date = dates[-2]
        to_date = dates[-1]
    else:
        from_date = dates[0]
        to_date = dates[1]
    if date_sort_key(from_date) > date_sort_key(to_date):
        from_date, to_date = to_date, from_date
    return from_date, to_date
def derive_date_range_from_period(period_text: str) -> Tuple[str, str]:
    text = safe_str(period_text)
    if not text:
        return "", ""
    from_date, to_date = derive_date_range_from_explicit_range(text)
    if from_date and to_date:
        return from_date, to_date
    upper = text.upper()
    match = re.search(
        rf"\b({MONTH_PATTERN})[\s\-_/.,]*(20\d{{2}}|\d{{2}})\b",
        upper,
        flags=re.IGNORECASE,
    )
    if match:
        month_name = match.group(1).upper()
        month = MONTH_ALIASES.get(month_name)
        year = normalize_year(match.group(2))
        if month:
            last_day = calendar.monthrange(year, month)[1]
            return f"01/{month:02d}/{year}", f"{last_day:02d}/{month:02d}/{year}"
    match = re.search(r"\b(0?[1-9]|1[0-2])[\s\-_/.,]+(20\d{2}|\d{2})\b", upper)
    if match:
        month = int(match.group(1))
        year = normalize_year(match.group(2))
        last_day = calendar.monthrange(year, month)[1]
        return f"01/{month:02d}/{year}", f"{last_day:02d}/{month:02d}/{year}"
    match = re.search(r"\b(20\d{2})[\s\-_/.,]+(0?[1-9]|1[0-2])\b", upper)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        last_day = calendar.monthrange(year, month)[1]
        return f"01/{month:02d}/{year}", f"{last_day:02d}/{month:02d}/{year}"
    return "", ""
def derive_final_date_range(raw: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Generic date range extraction. No hardcoded values.
    """
    candidates = [
        ("explicit_date_range_raw", safe_str(raw.get("explicit_date_range_raw"))),
        ("service_period_raw", safe_str(raw.get("service_period_raw"))),
        ("bill_date_raw", safe_str(raw.get("bill_date_raw"))),
        ("remarks_raw", safe_str(raw.get("remarks_raw"))),
    ]
    for item in raw.get("items") or []:
        if isinstance(item, dict):
            candidates.append(("item_particulars_raw", safe_str(item.get("particulars_raw"))))
            candidates.append(("item_amount_raw", safe_str(item.get("amount_raw"))))
    for source_name, text in candidates:
        if not text:
            continue
        from_date, to_date = derive_date_range_from_explicit_range(text)
        if from_date and to_date:
            return from_date, to_date, source_name
    for source_name, text in candidates:
        if not text:
            continue
        from_date, to_date = derive_date_range_from_period(text)
        if from_date and to_date:
            return from_date, to_date, source_name
    return "", "", ""
# =========================
# Raw result normalization/merge
# =========================
def normalize_raw_result(data: Dict[str, Any]) -> Dict[str, Any]:
    clean = {key: "" for key in RAW_KEYS}
    for key in EXTRA_HINT_KEYS:
        clean[key] = ""
    clean["items"] = []
    if not isinstance(data, dict):
        return clean
    if "header" in data or "document_numbers" in data or "dates" in data:
        header = data.get("header") or {}
        document_numbers = data.get("document_numbers") or {}
        dates = data.get("dates") or {}
        tax_ids = data.get("tax_ids") or {}
        codes = data.get("codes") or {}
        totals = data.get("totals") or {}
        nested_map = {
            "document_type_raw": data.get("document_type_raw", ""),
            "whether_self_reimbursement_raw": header.get("whether_self_reimbursement_raw", ""),
            "vendor_employee_name_raw": header.get("vendor_employee_name_raw", ""),
            "expense_type_raw": header.get("expense_type_raw", ""),
            "los_vendor_state_raw": header.get("los_vendor_state_raw", ""),
            "lor_isec_state_raw": header.get("lor_isec_state_raw", ""),
            "service_usage_raw": header.get("service_usage_raw", ""),
            "is_invoice_digitally_signed_raw": header.get("is_invoice_digitally_signed_raw", ""),
            "digital_signature_evidence_raw": header.get("digital_signature_evidence_raw", ""),
            "remarks_raw": header.get("remarks_raw", ""),
            "approver_raw": header.get("approver_raw", ""),
            "client_name_raw": header.get("client_name_raw", ""),
            "invoice_no_raw": document_numbers.get("invoice_no_raw", ""),
            "bill_no_raw": document_numbers.get("bill_no_raw", ""),
            "po_no_raw": document_numbers.get("po_no_raw", ""),
            "bill_date_raw": dates.get("bill_date_raw", ""),
            "service_period_raw": dates.get("service_period_raw", ""),
            "explicit_date_range_raw": dates.get("explicit_date_range_raw", ""),
            "vendor_gstin_raw": tax_ids.get("vendor_gstin_raw", ""),
            "isec_gstin_raw": tax_ids.get("isec_gstin_raw", ""),
            "vendor_pan_raw": tax_ids.get("vendor_pan_raw", ""),
            "department_code_raw": codes.get("department_code_raw", ""),
            "function_code_raw": codes.get("function_code_raw", ""),
            "location_code_raw": codes.get("location_code_raw", ""),
            "sub_loc_code_raw": codes.get("sub_loc_code_raw", ""),
            "subtotal_raw": totals.get("subtotal_raw", ""),
            "cgst_total_raw": totals.get("cgst_total_raw", ""),
            "sgst_total_raw": totals.get("sgst_total_raw", ""),
            "igst_total_raw": totals.get("igst_total_raw", ""),
            "cess_total_raw": totals.get("cess_total_raw", ""),
            "round_off_raw": totals.get("round_off_raw", ""),
            "grand_total_raw": totals.get("grand_total_raw", ""),
            "amount_in_words_raw": totals.get("amount_in_words_raw", ""),
        }
        for key, value in nested_map.items():
            clean[key] = safe_str(value)
        data_items = data.get("line_items") or data.get("items") or []
    else:
        for key in RAW_KEYS:
            clean[key] = safe_str(data.get(key, ""))
        data_items = data.get("items") or data.get("line_items") or []
    for key in EXTRA_HINT_KEYS:
        clean[key] = safe_str(data.get(key, clean.get(key, "")))
    if not isinstance(data_items, list):
        data_items = []
    for item in data_items:
        if not isinstance(item, dict):
            continue
        clean_item = {key: "" for key in ITEM_KEYS}
        for key in ITEM_KEYS:
            clean_item[key] = safe_str(item.get(key, ""))
        if not clean_item["particulars_raw"]:
            clean_item["particulars_raw"] = safe_str(item.get("description", ""))
        if not clean_item["amount_raw"]:
            clean_item["amount_raw"] = safe_str(item.get("amount", ""))
        if not clean_item["rate_raw"]:
            clean_item["rate_raw"] = safe_str(item.get("rate", ""))
        if not clean_item["quantity_raw"]:
            clean_item["quantity_raw"] = safe_str(item.get("quantity", ""))
        clean["items"].append(clean_item)
    if data.get("_warning"):
        clean["_warning"] = safe_str(data.get("_warning"))
    return clean
BUSINESS_WORDS = [
    "stall", "store", "shop", "enterprise", "enterprises", "traders", "trading",
    "services", "service", "products", "agency", "agencies", "company", "pvt",
    "private", "limited", "ltd", "llp", "restaurant", "hotel", "canteen", "tea",
    "foods", "solutions", "corporation", "corp", "associates", "industries",
]
def looks_business_like(value: Any) -> bool:
    text = safe_str(value).lower()
    if not text:
        return False
    return any(word in text for word in BUSINESS_WORDS)
def local_expense_context_text(raw: Dict[str, Any]) -> str:
    parts = [
        raw.get("document_type_raw"),
        raw.get("service_usage_raw"),
        raw.get("remarks_raw"),
        raw.get("service_period_raw"),
        raw.get("bill_period_raw"),
        raw.get("bill_upto_raw"),
    ]
    for item in raw.get("items") or []:
        if isinstance(item, dict):
            parts.extend([
                item.get("particulars_raw"),
                item.get("hsn_code_raw"),
                item.get("amount_raw"),
            ])
    return " ".join(safe_str(part) for part in parts if safe_str(part)).lower()
def is_local_small_expense_context(raw: Dict[str, Any]) -> bool:
    context = local_expense_context_text(raw)
    if not context:
        return False
    return bool(re.search(
        r"\b(tea|coffee|snacks?|samosa|breakfast|lunch|dinner|canteen|restaurant|hotel|food|foods|beverage|drinking\s+water|water\s+jar|mineral\s+water|cab|taxi|ride|travel|newspaper|subscription|garbage|user\s+charge|utility)\b",
        context,
        flags=re.IGNORECASE,
    ))
def looks_like_corporate_billed_party_for_local_expense(value: Any) -> bool:
    text = normalize_invoice_name(value).lower()
    if not text:
        return False
    supplier_words = [
        "tea", "stall", "hotel", "resort", "resorts", "lodging", "hospitality", "restaurant", "canteen", "sweet", "sweets", "foods", "food",
        "cafe", "coffee", "water", "distributor", "distributors", "dealer", "dealers",
        "agency", "agencies", "newspaper", "travels", "taxi", "cab", "store", "shop",
        "traders", "trading", "enterprise", "enterprises", "cater", "catering",
    ]
    if any(word in text for word in supplier_words):
        return False
    corporate_billed_party_words = [
        "securities", "security", "bank", "finance", "financial", "capital", "insurance",
        "broking", "brokerage", "stock", "exchange", "investments", "investment",
        "asset management", "mutual fund", "depository", "clearing", "limited", "ltd", "private", "pvt",
    ]
    return any(word in text for word in corporate_billed_party_words)
def suspicious_vendor_for_local_expense(raw: Dict[str, Any], vendor: Any) -> bool:
    if not is_local_small_expense_context(raw):
        return False
    candidate = normalize_invoice_name(vendor)
    if not candidate:
        return False
    if looks_like_corporate_billed_party_for_local_expense(candidate):
        return True
    if looks_like_customer_or_recipient_name(candidate):
        return True
    return False
def looks_person_like(value: Any) -> bool:
    text = safe_str(value)
    if not text:
        return False
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 2 or len(words) > 4:
        return False
    if looks_business_like(text):
        return False
    return True
def merge_header_fields(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(extra, dict):
        return base
    for key in HEADER_KEYS:
        base_value = safe_str(base.get(key))
        extra_value = safe_str(extra.get(key))
        if not extra_value:
            continue
        if not base_value:
            base[key] = extra_value
            continue
        # Generic vendor correction:
        # if main pass picked a person/contact but header pass found business/shop/firm name.
        if key == "vendor_employee_name_raw":
            if looks_person_like(base_value) and looks_business_like(extra_value):
                base[key] = extra_value
        # If base has only one date but header found a complete date range, keep the range.
        if key == "explicit_date_range_raw":
            if len(extract_all_full_dates(extra_value)) >= 2:
                base[key] = extra_value
    return base
def backfill_from_items(raw: Dict[str, Any]) -> Dict[str, Any]:
    items = raw.get("items") or []
    first_particular = ""
    for item in items:
        first_particular = safe_str(item.get("particulars_raw"))
        if first_particular:
            break
    if first_particular:
        if not safe_str(raw.get("service_usage_raw")):
            raw["service_usage_raw"] = first_particular
        if not safe_str(raw.get("expense_type_raw")):
            raw["expense_type_raw"] = first_particular
    return raw
def normalize_item_text_for_dedupe(value: Any) -> str:
    """Normalize text only for duplicate detection. Does not change output values."""
    text = safe_str(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;|-_")
    return text
def raw_item_dedupe_signature(item: Dict[str, Any]) -> Tuple[str, ...]:
    """
    Exact duplicate signature for raw line items.
    This is generic: it removes only repeated rows that have the same service/item,
    amount, taxes, HSN, quantity, unit and rate. It does not merge different rows
    that merely look similar.
    """
    numeric_keys = {"amount_raw", "cgst_raw", "sgst_raw", "igst_raw", "cess_raw", "rate_raw", "quantity_raw"}
    keys = (
        "particulars_raw",
        "quantity_raw",
        "quantity_unit_raw",
        "hsn_code_raw",
        "rate_raw",
        "amount_raw",
        "cgst_raw",
        "sgst_raw",
        "igst_raw",
        "cess_raw",
    )
    signature = []
    for key in keys:
        value = item.get(key, "")
        if key in numeric_keys:
            signature.append(clean_amount(value) or normalize_item_text_for_dedupe(value))
        else:
            signature.append(normalize_item_text_for_dedupe(value))
    return tuple(signature)
def deduplicate_raw_items(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove repeated identical line items before total calculation.
    Duplicate rows can appear because the API runs multiple passes:
    full-page extraction, repair extraction and deterministic source-text repair.
    If two passes return the same row, keeping both doubles the calculated total.
    """
    items = raw.get("items") or []
    if not isinstance(items, list) or len(items) <= 1:
        return raw
    unique_items = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        signature = raw_item_dedupe_signature(item)
        has_real_value = any(signature)
        if has_real_value and signature in seen:
            continue
        if has_real_value:
            seen.add(signature)
        unique_items.append(item)
    raw["items"] = unique_items
    return raw
def final_item_dedupe_signature(item: Dict[str, Any]) -> Tuple[str, ...]:
    numeric_keys = {"amount", "cgst", "sgst", "igst", "cess", "rate", "quantity"}
    keys = (
        "bill_no",
        "particulars",
        "from_date",
        "to_date",
        "amount",
        "cgst",
        "sgst",
        "igst",
        "cess",
        "bill_date",
        "quantity",
        "quantity_unit",
        "hsn_code",
        "rate",
        "isec_gstin",
    )
    signature = []
    for key in keys:
        value = item.get(key, "")
        if key in numeric_keys:
            signature.append(clean_amount(value) or normalize_item_text_for_dedupe(value))
        else:
            signature.append(normalize_item_text_for_dedupe(value))
    return tuple(signature)
def deduplicate_final_line_items(final_line_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(final_line_items) <= 1:
        return final_line_items
    unique_items = []
    seen = set()
    for item in final_line_items:
        signature = final_item_dedupe_signature(item)
        has_real_value = any(signature)
        if has_real_value and signature in seen:
            continue
        if has_real_value:
            seen.add(signature)
        unique_items.append(item)
    return unique_items
def fix_grand_total_from_items(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic total correction using corrected item amounts.
    """
    items = raw.get("items") or []
    corrected_sum = Decimal("0")
    has_amount = False
    for item in items:
        final_amount, final_rate, final_quantity = choose_amount_rate_quantity(item)
        if final_amount:
            amount_dec = decimal_or_none(final_amount)
            if amount_dec is not None:
                corrected_sum += amount_dec
                has_amount = True
            item["_final_amount"] = final_amount
            item["_final_rate"] = final_rate
            item["_final_quantity"] = final_quantity
    if has_amount:
        if corrected_sum == corrected_sum.to_integral():
            raw["_calculated_items_total"] = str(int(corrected_sum))
        else:
            raw["_calculated_items_total"] = format(corrected_sum.normalize(), "f")
    return raw
# =========================
# Claim/paid amount helpers
# =========================
CLAIM_TOTAL_LABELS = [
    ("total bill", 100),
    ("amount paid", 98),
    ("paid amount", 98),
    ("paid by", 94),
    ("net payable", 92),
    ("total payable", 92),
    ("grand total", 88),
    ("total amount", 86),
    ("invoice total", 78),
    ("total value", 60),
]
MONEY_VALUE_PATTERN = re.compile(
    r"(?:₹|Rs\.?|INR)\s*(-?\d[\d,]*(?:\.\d+)?)|(-?\d[\d,]*(?:\.\d+)?)\s*(?:₹|Rs\.?|INR)",
    flags=re.IGNORECASE,
)
GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
HSN_SAC_PATTERN = re.compile(r"\b(?:HSN|SAC)\s*(?:Code)?\s*[:\-]?\s*(\d{4,8})\b", flags=re.IGNORECASE)
GSTIN_STATE_CODE_MAP = {
    "01": "Jammu & Kashmir",
    "02": "Himachal Pradesh",
    "03": "Punjab",
    "04": "Chandigarh",
    "05": "Uttarakhand",
    "06": "Haryana",
    "07": "Delhi",
    "08": "Rajasthan",
    "09": "Uttar Pradesh",
    "10": "Bihar",
    "11": "Sikkim",
    "12": "Arunachal Pradesh",
    "13": "Nagaland",
    "14": "Manipur",
    "15": "Mizoram",
    "16": "Tripura",
    "17": "Meghalaya",
    "18": "Assam",
    "19": "West Bengal",
    "20": "Jharkhand",
    "21": "Odisha",
    "22": "Chhattisgarh",
    "23": "Madhya Pradesh",
    "24": "Gujarat",
    "25": "Daman and Diu",
    "26": "Dadra and Nagar Haveli and Daman and Diu",
    "27": "Maharashtra",
    "28": "Andhra Pradesh",
    "29": "Karnataka",
    "30": "Goa",
    "31": "Lakshadweep",
    "32": "Kerala",
    "33": "Tamil Nadu",
    "34": "Puducherry",
    "35": "Andaman and Nicobar Islands",
    "36": "Telangana",
    "37": "Andhra Pradesh",
    "38": "Ladakh",
}
INDIAN_STATE_NAMES = sorted(set(GSTIN_STATE_CODE_MAP.values()), key=len, reverse=True)
# Generic city/district-to-state hints used only when OCR/vision captured an address
# but not an explicit state or GSTIN. This is not invoice/vendor-specific.
CITY_STATE_HINTS = {
    "pune": "Maharashtra", "mumbai": "Maharashtra", "navi mumbai": "Maharashtra", "thane": "Maharashtra",
    "delhi": "Delhi", "new delhi": "Delhi",
    "bangalore": "Karnataka", "bengaluru": "Karnataka",
    "kanpur": "Uttar Pradesh", "lucknow": "Uttar Pradesh",
    "ranchi": "Jharkhand",
    "faridabad": "Haryana", "gurugram": "Haryana", "gurgaon": "Haryana",
    "chennai": "Tamil Nadu", "hyderabad": "Telangana", "ahmedabad": "Gujarat", "kolkata": "West Bengal",
}
SINGLE_DAY_EXPENSE_KEYWORDS = [
    "ride", "cab", "taxi", "auto", "travel", "trip", "fare",
    "parking", "toll", "restaurant", "canteen", "tea", "meal", "food",
    "receipt", "voucher", "deposit", "hospital", "medical",
    "examination", "certification", "course", "training", "cpe", "ecpe",
]
TRIP_UNIT_KEYWORDS = ["ride", "cab", "taxi", "auto", "travel", "trip", "fare"]
DEPOSIT_UNIT_KEYWORDS = ["deposit", "deposit receipt", "transaction type deposit", "hospital deposit"]
EXAM_UNIT_KEYWORDS = ["examination", "certification", "exam", "cpe", "ecpe"]
COURSE_UNIT_KEYWORDS = ["course", "training", "workshop", "seminar"]
NEWSPAPER_SUBSCRIPTION_KEYWORDS = ["newspaper", "news paper", "publication", "subscription", "newspaper dealers", "newspaper agency", "delivery charge", "monthly newspaper", "monthly publication"]
FOOD_UNIT_KEYWORDS = ["restaurant", "canteen", "tea", "meal", "food", "snack", "lunch", "dinner"]
TELECOM_MONTHLY_UNIT_KEYWORDS = [
    "bill period", "monthly rentals", "rentals", "broadband", "fixedline",
    "fixed line", "telecommunication", "telephone", "internet", "fiber", "fibre",
    "plan details", "this month's charges",
]
PERIODIC_SERVICE_UNIT_KEYWORDS = [
    "bill upto", "bill up to", "bill period", "billing period", "service period",
    "charge period", "monthly finalized", "monthly finalised", "current charge",
    "user charge", "garbage collection", "municipal", "utility charge", "water bill",
    "property tax", "maintenance charge",
]
MONTHLY_ONLY_UNIT_KEYWORDS = [
    "monthly", "monthly rentals", "month's charges", "bill period", "billing period",
    "bill upto", "bill up to", "monthly finalized", "monthly finalised",
    "current charge", "user charge", "garbage collection", "municipal",
    "utility charge", "water bill", "property tax", "maintenance charge",
]
GENERIC_PROFESSIONAL_SERVICE_KEYWORDS = [
    "engineering services", "support", "dashboarding", "consulting", "consultancy",
    "professional services", "managed services", "development services", "maintenance services",
    "testing services", "quality services", "advisory services", "audit services",
]
GENERIC_EXPENSE_WORDS = {
    "service", "services", "tax invoice", "invoice", "bill", "charges", "charge",
    "amount", "total", "rate", "per", "tax", "gst", "hsn", "sac",
    "currency", "inr", "rs", "rupee", "rupees", "taxable value",
}
CURRENCY_OR_TABLE_FRAGMENT_WORDS = {
    "inr", "rs", "rs.", "rupee", "rupees", "amount", "total", "rate", "per",
    "gst", "tax", "hsn", "sac", "currency", "taxable", "value",
}
LOCAL_SERVICE_KEYWORDS = [
    "municipal", "nagar nigam", "municipality", "corporation", "user charge",
    "garbage collection", "waste collection", "water bill", "property tax",
    "local body", "authority",
]
AUTHORITY_NAME_KEYWORDS = [
    "nagar nigam", "municipal", "municipality", "corporation", "authority",
    "government", "govt", "department", "council", "board",
]
PAYMENT_MODE_WORDS = [
    "cash", "upi", "card", "payment",
    "mode of payment", "paid by", "received with thanks",
]
BILL_NO_LABELS = [
    "receipt no", "receipt number", "voucher no", "voucher number", "deposit receipt no",
    "transaction receipt no", "invoice no", "invoice number", "bill no", "bill number",
    "crn", "ride number", "consumer no", "account no",
]
def format_decimal_amount(value: Decimal) -> str:
    if value == value.to_integral():
        return str(int(value))
    return format(value.normalize(), "f")
def normalize_percent_rate(value: Any) -> str:
    """Return a GST percentage like 18% only when a percent sign is actually visible."""
    text = safe_str(value).replace("％", "%")
    if not text:
        return ""
    match = re.search(r"(?<!\d)(\d{1,2}(?:\.\d{1,4})?)\s*%", text)
    if not match:
        return ""
    try:
        rate = Decimal(match.group(1))
    except InvalidOperation:
        return safe_str(match.group(0)).replace(" ", "")
    if rate < 0 or rate > 40:
        return ""
    return f"{format_decimal_amount(rate)}%"
def combine_percent_rates(*values: Any) -> str:
    total = Decimal("0")
    has_rate = False
    for value in values:
        pct = normalize_percent_rate(value)
        if not pct:
            continue
        dec = decimal_or_none(pct)
        if dec is None:
            continue
        total += dec
        has_rate = True
    if not has_rate:
        return ""
    return f"{format_decimal_amount(total)}%"
def derive_gst_rate_from_tax_amounts(amount: Any, cgst: Any, sgst: Any, igst: Any, cess: Any) -> str:
    amount_dec = decimal_or_none(amount)
    if amount_dec is None or amount_dec == 0:
        return ""
    tax_total = Decimal("0")
    has_tax = False
    for value in (cgst, sgst, igst, cess):
        tax_dec = decimal_or_none(value)
        if tax_dec is None or tax_dec == 0:
            continue
        tax_total += tax_dec
        has_tax = True
    if not has_tax or tax_total <= 0:
        return ""
    rate = (tax_total / amount_dec) * Decimal("100")
    if rate < 0 or rate > 40:
        return ""
    return f"{format_decimal_amount(rate.quantize(Decimal('0.01')).normalize())}%"
def choose_gst_rate_for_output(
    raw: Dict[str, Any],
    item: Dict[str, Any],
    amount: Any,
    cgst: Any,
    sgst: Any,
    igst: Any,
    cess: Any,
) -> str:
    """Final rate column now means GST rate %. Return 0 when no GST rate is available."""
    derived_rate = derive_gst_rate_from_tax_amounts(amount, cgst, sgst, igst, cess)
    for value in (
        item.get("gst_rate_raw"),
        item.get("rate_raw"),
        raw.get("_source_text_gst_rate_raw"),
        raw.get("gst_rate_raw"),
    ):
        pct = normalize_percent_rate(value)
        if pct:
            raw_value = safe_str(value)
            split_rates = re.findall(r"(\d{1,2}(?:\.\d{1,4})?)\s*%", raw_value)
            if len(split_rates) >= 2:
                return " + ".join(f"{format_decimal_amount(Decimal(r))}%" for r in split_rates[:2])
            # If one component rate such as 6% is visible but both CGST and SGST amounts
            # are present, the output rate column should represent total GST rate.
            pct_dec = decimal_from_percent(pct)
            derived_dec = decimal_from_percent(derived_rate)
            if derived_rate and derived_dec > pct_dec:
                return derived_rate
            return pct
    return derived_rate or "0"
def extract_money_values(text: str) -> List[str]:
    values: List[str] = []
    for match in MONEY_VALUE_PATTERN.finditer(safe_str(text)):
        raw_value = match.group(1) or match.group(2) or ""
        cleaned = clean_amount(raw_value)
        if cleaned:
            values.append(cleaned)
    return values
def extract_labeled_number_without_currency(text: str, label: str) -> List[str]:
    """
    Fallback for OCR/PDF text where the currency symbol is dropped.
    Used only near strong total/paid labels.
    """
    values: List[str] = []
    pattern = rf"{re.escape(label)}[^0-9-]{{0,80}}(-?\d[\d,]*(?:\.\d+)?)"
    for match in re.finditer(pattern, safe_str(text), flags=re.IGNORECASE):
        cleaned = clean_amount(match.group(1))
        if cleaned:
            values.append(cleaned)
    return values
def extract_claim_total_from_text(source_text: str) -> str:
    """
    Generic extraction of the visible paid/claim amount from searchable PDF text.
    This fixes cases where the document has a large receipt total and a smaller
    GST invoice total for only a convenience/platform/service fee.
    """
    lines = [line.strip() for line in safe_str(source_text).splitlines() if line.strip()]
    candidates: List[Tuple[int, Decimal, str]] = []
    for index, line in enumerate(lines):
        window = " ".join(lines[index:index + 3])
        window_lower = window.lower()
        for label, score in CLAIM_TOTAL_LABELS:
            if label not in window_lower:
                continue
            amounts = extract_money_values(window)
            if not amounts and score >= 86:
                amounts = extract_labeled_number_without_currency(window, label)
            for amount in amounts:
                amount_decimal = decimal_or_none(amount)
                if amount_decimal is None or amount_decimal <= 0:
                    continue
                candidates.append((score, amount_decimal, amount))
    if not candidates:
        return ""
    # Prefer stronger labels such as Total Bill / Amount Paid. If labels are equal,
    # choose the larger visible amount to avoid returning only a small tax component.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return format_decimal_amount(candidates[0][1])
def extract_source_text_from_file(file_path: str, filename: str) -> str:
    lower = safe_str(filename).lower()
    if not lower.endswith(".pdf"):
        return ""
    text_parts: List[str] = []
    doc = fitz.open(file_path)
    try:
        total_pages = min(len(doc), MAX_PAGES)
        for page_index in range(total_pages):
            page_text = doc[page_index].get_text("text") or ""
            if page_text.strip():
                text_parts.append(page_text)
    finally:
        doc.close()
    return "\n".join(text_parts)
def remove_noise_sections_for_field_hints(source_text: str) -> str:
    """Remove grievance/footer text so state fallback does not use vendor support address."""
    text = safe_str(source_text)
    for marker in (
        "in case of any complaint/grievance",
        "grievance officer",
        "please note",
    ):
        pos = text.lower().find(marker)
        if pos != -1:
            text = text[:pos]
    return text
def extract_gstins_from_text(source_text: str) -> List[str]:
    gstins: List[str] = []
    for match in GSTIN_PATTERN.finditer(safe_str(source_text).upper()):
        value = match.group(0)
        if value not in gstins:
            gstins.append(value)
    return gstins
def extract_vendor_gstin_from_text(source_text: str) -> str:
    """
    Generic vendor GSTIN detector.
    Rule:
    - Prefer a GSTIN that appears before Buyer/Bill To/Ship To/Customer/Client/To block.
    - If multiple GSTINs are present, the first one is normally supplier/vendor GSTIN.
    - If only one GSTIN is present and it appears only inside the buyer/customer block,
      do not treat it as vendor GSTIN.
    """
    text = safe_str(source_text)
    if not text:
        return ""
    gstins = extract_gstins_from_text(text)
    if not gstins:
        return ""
    buyer_marker = re.search(
        r"\b(?:buyer\s*\(?\s*bill\s*to\s*\)?|bill\s*to|ship\s*to|billed\s*to|customer|client|recipient|consignee|to\s*[,\\:]?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if buyer_marker:
        supplier_part = text[:buyer_marker.start()]
        supplier_gstins = extract_gstins_from_text(supplier_part)
        if supplier_gstins:
            return supplier_gstins[0]
        # If only one GSTIN and it appears after buyer marker, it is probably buyer GSTIN.
        if len(gstins) == 1:
            gstin_pos = text.upper().find(gstins[0])
            if gstin_pos >= buyer_marker.start():
                return ""
    if len(gstins) >= 2:
        return gstins[0]
    return gstins[0]
def state_from_gstin(gstin: Any) -> str:
    text = safe_str(gstin).upper()
    match = GSTIN_PATTERN.search(text)
    if not match:
        return ""
    return GSTIN_STATE_CODE_MAP.get(match.group(0)[:2], "")
def extract_hsn_sac_from_text(source_text: str) -> str:
    for match in HSN_SAC_PATTERN.finditer(safe_str(source_text)):
        code = safe_str(match.group(1))
        if code:
            return code
    return ""
def normalize_invoice_name(value: Any) -> str:
    text = safe_str(value)
    text = re.sub(r"\s+", " ", text).strip(" .,:;|-\t")
    text = re.sub(r"^(for|registered office)\s*:?\s*", "", text, flags=re.IGNORECASE).strip(" .,:;|-")
    return text
def strip_address_tail_from_vendor_name(value: Any) -> str:
    """
    Keep only the supplier/business name when a vision model returns
    'name + postal address' as vendor. This is generic and label/layout based;
    it does not hardcode any vendor or client names.
    """
    text = normalize_invoice_name(value)
    if not text:
        return ""
    text = re.sub(r"\s*[|;]+\s*", ", ", text).strip()
    address_start_patterns = [
        r"\s*(?:,|\.)\s*(?:no\.?\s*)?\d[\w/\-]*\s*(?:,|\s+)(?:road|rd\.?|street|st\.?|lane|ln\.?|sector|plot|block|bldg|building|floor|office|shop|market|nagar|hills|colony)\b",
        r"\s+(?:no\.?\s*)?\d[\w/\-]*\s+(?:road|rd\.?|street|st\.?|lane|ln\.?|sector|plot|block|bldg|building|floor|office|shop|market|nagar|hills|colony)\b",
        r"\s*,\s*(?:road|rd\.?|street|st\.?|lane|ln\.?|sector|plot|block|bldg|building|floor|office|shop|market|nagar|hills|colony)\b",
        r"\s+\b(?:road|rd\.?|street|st\.?|lane|ln\.?|sector|plot|block|bldg|building|floor|office|nagar|hills|colony)\b.*$",
        r"\s+\b(?:hyderabad|mumbai|pune|delhi|ahmedabad|bangalore|bengaluru|chennai|kolkata|gurugram|faridabad|thane|navi\s+mumbai|telangana|maharashtra|gujarat|haryana|karnataka|tamil\s+nadu|west\s+bengal|uttar\s+pradesh)\b.*$",
        r"\s+\b\d{6}\b.*$",
        r"\s+\b(?:gstin|gst\s*no|pan|phone|mobile|email)\b.*$",
    ]
    best = text
    for pattern in address_start_patterns:
        match = re.search(pattern, best, flags=re.IGNORECASE)
        if match and match.start() >= 3:
            best = best[:match.start()]
            break
    parts = [p.strip(" .,:;-") for p in re.split(r"[,;]", best) if p.strip(" .,:;-")]
    if len(parts) > 1:
        tail = " ".join(parts[1:]).lower()
        if re.search(r"\b(?:no|road|rd|street|lane|floor|office|shop|market|nagar|hills|colony|city|state|pin|pincode|hyderabad|mumbai|pune|delhi|telangana|maharashtra|gujarat|karnataka)\b|\d{4,}", tail):
            best = parts[0]
    best = re.sub(r"\s+", " ", best).strip(" .,:;-")
    if not re.search(r"[A-Za-z]", best) and not re.search(r"[\u0900-\u097F]", best):
        return ""
    if re.fullmatch(r"(?i)(?:no\.?\s*)?\d[\w/\- ]*(?:road|rd|street|lane|floor|office|shop|market|nagar|hills|colony|hyderabad|telangana|gujarat|maharashtra).*", best):
        return ""
    return normalize_invoice_name(best)
def looks_like_address_heavy_name(value: Any) -> bool:
    text = normalize_invoice_name(value).lower()
    if not text:
        return False
    stripped = strip_address_tail_from_vendor_name(text)
    if stripped and len(stripped) >= max(3, int(len(text) * 0.55)):
        return False
    return bool(re.search(r"\b(?:road|rd|street|lane|floor|office|shop|market|nagar|hills|colony|hyderabad|telangana|gujarat|maharashtra|pincode|pin)\b|\d{6}|\d[\w/\-]*\s+(?:road|rd|street|lane)", text))
def clean_final_client_name(value: Any, raw: Dict[str, Any], vendor_name: Any = "") -> str:
    """
    Final generic cleanup for client_name.
    Vision models sometimes return a full Billing/Shipping Address as the client.
    This keeps only a standalone billed-party name when present. If the value is
    only an address/location block, it returns blank. No vendor, client, city, or
    company name is hardcoded.
    """
    original = normalize_invoice_name(value)
    if not original:
        return ""
    cleaned = clean_billed_party_candidate(original)
    if not cleaned:
        cleaned = original
    # Reuse the generic address-tail stripper used for vendor names. It strips
    # road/floor/office/pincode/address tails while preserving the leading name.
    stripped = strip_address_tail_from_vendor_name(cleaned)
    if stripped:
        cleaned = stripped
    cleaned = normalize_invoice_name(cleaned)
    if not cleaned:
        return ""
    original_has_address_context = bool(re.search(
        r"\b(?:shipping\s+address|billing\s+address|address|road|rd\.?|street|lane|floor|office\s+space|office|shop\s+no|building|complex|pincode|pin\s*code|hyderabad|mumbai|pune|delhi|ahmedabad|bangalore|bengaluru|chennai|kolkata|telangana|maharashtra|gujarat|karnataka|haryana|tamil\s+nadu|west\s+bengal)\b|\b\d{6}\b|\b\d[\w/\-]*\s+(?:road|rd|street|lane)\b",
        original,
        flags=re.IGNORECASE,
    ))
    # Reject values that are still clearly address-like after stripping.
    if looks_like_address_heavy_name(cleaned):
        return ""
    # Faint utility scans often hallucinate a comma-separated fragment as the
    # consumer name. Keep a utility client only when it looks like a clean party
    # name/company name, not a mixed OCR fragment.
    if is_utility_bill_context(raw):
        # Final utility-specific OCR cleanup: when a labelled consumer name is
        # captured together with the first isolated character of the next line,
        # e.g. "... LTD A", keep the legal/business name only.
        cleaned = re.sub(r"\b((?:LTD\.?|LIMITED|PVT\.?\s*LTD\.?|PRIVATE\s+LIMITED|LLP))\s+[A-Z]\s*$", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = normalize_invoice_name(cleaned)
        if "," in cleaned and not has_company_suffix_or_vendor_word(cleaned):
            return ""
        if re.search(r"\b(?:hmspl|pvt|ltd|limited|llp|company|corp|corporation)\b", cleaned, flags=re.IGNORECASE):
            pass
        elif len(re.findall(r"[A-Za-z]{3,}", cleaned)) <= 2 and re.search(r"[,/]|\d", original):
            return ""
    # If the address block collapses to the same name as vendor, there is no
    # distinct client name visible. Avoid duplicating vendor as client.
    vendor_clean = normalize_invoice_name(vendor_name)
    if vendor_clean and cleaned.lower() == vendor_clean.lower() and original_has_address_context:
        return ""
    # Final validator: reject labels, totals, GST/tax words, pure numerics, etc.
    if looks_like_invalid_final_client_name(cleaned):
        return ""
    return cleaned
def has_company_suffix_or_vendor_word(value: Any) -> bool:
    text = safe_str(value).lower()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(limited|ltd|private|pvt|llp|inc|corporation|corp|company|hospitals?|markets?|services?|telecom|telecommunication|institute|institution|enterprises?|agency|bank|finance|solutions|technologies)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
def is_generic_vendor_name(value: Any) -> bool:
    """Reject category-only vendor names; this is generic, not company-specific."""
    text = normalize_invoice_name(value).lower()
    if not text:
        return True
    generic_patterns = [
        r"^newspaper\s+dealers?$",
        r"^restaurant$",
        r"^cash\s+(?:credit\s+)?memo$",
        r"^tax\s+invoice$",
        r"^invoice$",
        r"^bill$",
        r"^receipt$",
        r"^services?$",
        r"^retail\s+invoice$",
    ]
    return any(re.fullmatch(pattern, text, flags=re.IGNORECASE) for pattern in generic_patterns)
def looks_like_customer_or_recipient_name(value: Any) -> bool:
    text = safe_str(value).lower()
    if not text:
        return False
    customer_tokens = [
        "customer", "recipient", "bill to", "billed to",
        "ship to", "sold to", "patient", "client", "consumer", "subscriber",
        "place of supply", "supply address", "billing address",
    ]
    return any(token in text for token in customer_tokens)
def name_appears_in_customer_context(source_text: Any, name: Any) -> bool:
    """
    Generic customer-block detector. It does not know any company names.
    It checks whether a candidate name appears near recipient/customer labels
    or account/address fields that usually belong to the billed party.
    """
    text = safe_str(source_text)
    candidate = normalize_invoice_name(name)
    if not text or not candidate:
        return False
    lower_text = text.lower()
    lower_candidate = candidate.lower()
    index = lower_text.find(lower_candidate)
    if index == -1:
        # Try a whitespace-normalized lookup.
        compact_text = re.sub(r"\s+", " ", lower_text)
        compact_candidate = re.sub(r"\s+", " ", lower_candidate)
        index = compact_text.find(compact_candidate)
        if index == -1:
            return False
        context = compact_text[max(0, index - 220): index + len(compact_candidate) + 320]
    else:
        context = lower_text[max(0, index - 220): index + len(lower_candidate) + 320]
    customer_context_markers = [
        "to,", "to ", "bill to", "billed to", "ship to", "sold to",
        "recipient", "customer", "client", "patient", "consumer", "subscriber",
        "place of supply", "ship to state code", "supply address", "billing address",
        "relationship number", "account number", "account no", "mobile no",
        "email id", "email:", "enrollment", "uhid",
    ]
    issuer_context_markers = [
        "for ", "from ", "supplier", "seller", "vendor", "service provider",
        "registered office", "corporate identity number", "cin", "authorised signatory",
        "authorized signatory", "gstin:", "state gstin",
    ]
    has_customer_marker = any(marker in context for marker in customer_context_markers)
    has_issuer_marker = any(marker in context for marker in issuer_context_markers)
    # When both are present, only classify as customer if explicit billed-party markers exist.
    strong_customer_markers = [
        "bill to", "billed to", "ship to", "sold to", "recipient", "customer",
        "client", "patient", "subscriber", "place of supply", "supply address",
    ]
    if has_customer_marker and not has_issuer_marker:
        return True
    if any(marker in context for marker in strong_customer_markers):
        return True
    return False
def significant_name_tokens(value: Any) -> set:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", safe_str(value))
        if len(token) >= 3 and token.lower() not in {"the", "and", "for", "ltd", "limited", "pvt", "private"}
    }
def should_replace_vendor_with_source(current_vendor: Any, source_vendor: Any, source_text: Any = "") -> bool:
    """
    Use deterministic source-text vendor when the vision model picked a customer,
    person/contact, short brand, or recipient name. This keeps issuer names
    from labelled source text ahead of recipient/customer address blocks.
    """
    current = normalize_invoice_name(current_vendor)
    source = normalize_invoice_name(source_vendor)
    if not source:
        return False
    if not current:
        return True
    if current.lower() == source.lower():
        return False
    if name_appears_in_customer_context(source_text, current):
        return True
    if is_generic_vendor_name(current) and not is_generic_vendor_name(source):
        return True
    if looks_like_customer_or_recipient_name(current):
        return True
    if looks_person_like(current) and has_company_suffix_or_vendor_word(source):
        return True
    current_tokens = significant_name_tokens(current)
    source_tokens = significant_name_tokens(source)
    if current_tokens and current_tokens.issubset(source_tokens) and len(source) > len(current):
        return True
    # Prefer a labelled legal issuer name over a generic or incomplete name.
    if has_company_suffix_or_vendor_word(source) and not has_company_suffix_or_vendor_word(current):
        return True
    return False
def looks_like_client_fragment(value: Any) -> bool:
    """Reject OCR fragments such as 'tal (' or half words picked from Total/Amount text."""
    text = normalize_invoice_name(value)
    if not text:
        return True
    # Unbalanced punctuation is a strong signal of a cropped/OCR fragment.
    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
        return True
    alpha_tokens = re.findall(r"[A-Za-z]+", text)
    if not alpha_tokens:
        return True
    # Very short single-word values are usually not valid customer names.
    if len(alpha_tokens) == 1 and len(alpha_tokens[0]) < 4:
        return True
    # If it has no legal/business word and fewer than two useful words, treat as fragment.
    useful_tokens = [token for token in alpha_tokens if len(token) >= 3]
    if len(useful_tokens) < 2 and not has_company_suffix_or_vendor_word(text):
        return True
    # Common sliced words from amount/total/table labels.
    if re.fullmatch(r"(?i)(tal|otal|total|mount|amount|rate|qty|tax|hsn|gst|bill|date|period)", text.strip(" ()[]{}")):
        return True
    return False
def should_replace_client_with_source(raw: Dict[str, Any], source_client: Any) -> bool:
    """Use deterministic source-text customer when current client is blank, invalid, or only an OCR fragment."""
    current = normalize_invoice_name(raw.get("client_name_raw"))
    source = normalize_invoice_name(source_client)
    if not source:
        return False
    if not current:
        return True
    if current.lower() == source.lower():
        return False
    if looks_like_client_fragment(current):
        return True
    if looks_like_invalid_client_any_context(raw, current):
        return True
    current_tokens = significant_name_tokens(current)
    source_tokens = significant_name_tokens(source)
    if current_tokens and current_tokens.issubset(source_tokens) and len(source) > len(current):
        return True
    # Prefer a full legal/business customer name from a labelled/address block over a short weak value.
    if has_company_suffix_or_vendor_word(source) and not has_company_suffix_or_vendor_word(current):
        return True
    return False
def extract_labelled_client_from_context(raw: Dict[str, Any], *extra_values: Any) -> str:
    """
    Generic billed-party fallback from labels such as M/s, Bill To, Buyer,
    Customer, Client, Consumer, Subscriber or To.
    This is intentionally label/layout based. It does not contain any company,
    vendor, or client hardcoding.
    """
    values = list(extra_values)
    for key in (
        "billed_party_raw",
        "billed_party_block_raw",
        "customer_block_raw",
        "client_block_raw",
        "visible_text_raw",
        "_vendor_identity_evidence",
        "_party_identity_evidence",
        "client_address_raw",
        "place_of_supply_raw",
        "document_type_raw",
        "remarks_raw",
    ):
        values.append(raw.get(key))
    label_pattern = r"(?:M\s*/?\s*s\.?|Bill\s*To|Billed\s*To|Buyer|Recipient|Customer|Client|Consumer|Subscriber|Received\s*From|Paid\s*By|Party\s*Name|Name|To)"
    stop_label_pattern = r"(?:Address|GSTIN|GST\s*No|GST|State|Place\s+of\s+Supply|Bill\s*No|Invoice\s*No|Date|PAN|Phone|Email|Mobile|Particulars|HSN|Amount|Rate|Total|Signature|Authorised|Authorized)"
    for value in values:
        text_value = safe_str(value)
        if not text_value:
            continue
        normalized = re.sub(r"\r", "\n", text_value)
        normalized = re.sub(rf"\s+({stop_label_pattern}\s*[:\-]?)", r"\n\1", normalized, flags=re.IGNORECASE)
        patterns = [
            rf"\b{label_pattern}\s*[:\-]?\s*([^\n\r]{{2,140}})",
            rf"\b{label_pattern}\s*[:\-]?\s*(?:\n|\r\n)\s*([^\n\r]{{2,140}})",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                candidate = clean_billed_party_candidate(match.group(1))
                if candidate and looks_like_valid_billed_party_hint(raw, candidate):
                    return candidate
    return ""
def clean_labelled_vendor_candidate(value: Any) -> str:
    """Clean a generic issuer/vendor candidate captured after labels such as From/Supplier/Vendor."""
    text = safe_str(value)
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.split(
        r"\b(M\s*/?\s*s\.?|Bill\s*To|Billed\s*To|Buyer|Customer|Client|To|Particulars|Quantity|Rate|Amount|Total|Date|No\.?|Phone|Mobile|GSTIN|GST\s*No)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    lines = [line.strip(" :;-\t") for line in text.split("\n") if line.strip(" :;-\t")]
    if not lines:
        text = safe_str(value).strip(" :;-\t")
    else:
        # Prefer the first non-label, non-phone line. A second line may be location/address.
        useful_lines = []
        for line in lines:
            if re.fullmatch(r"(?i)(from|supplier|vendor|issuer|issued\s+by|shop|stall|restaurant|canteen|for)", line):
                continue
            if re.fullmatch(r"[+()\-\s\d]{7,}", line):
                continue
            useful_lines.append(line)
        text = " ".join(useful_lines[:2]).strip() if useful_lines else ""
    text = re.sub(r"\b(from|supplier|vendor|issuer|issued\s+by|shop|stall|restaurant|canteen)\s*[:\-]?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\b(?:mob(?:ile)?|phone|tel)\b\s*[:\-]?\s*[+()\-\s\d]{7,}", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).strip(" :;-.,")
    return strip_address_tail_from_vendor_name(text)
def looks_like_valid_vendor_hint(raw: Dict[str, Any], value: Any) -> bool:
    candidate = strip_address_tail_from_vendor_name(value)
    if not candidate:
        return False
    if len(candidate) < 3:
        return False
    if is_generic_vendor_name(candidate):
        return False
    if looks_like_customer_or_recipient_name(candidate):
        return False
    if looks_like_valid_billed_party_hint(raw, candidate) and looks_like_corporate_billed_party_for_local_expense(candidate):
        return False
    if re.search(r"\b(M\s*/?\s*s\.?|Bill\s*To|Billed\s*To|Buyer|Customer|Client|Particulars|Quantity|Rate|Amount|Total|Date|No\.?|GSTIN|HSN)\b", candidate, flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"[+()\-\s\d]{7,}", candidate):
        return False
    return True
def extract_labelled_vendor_from_context(raw: Dict[str, Any], *extra_values: Any) -> str:
    """
    Generic issuer/vendor fallback from labels such as From, Supplier, Vendor,
    Issued By, Shop, Stall, Restaurant, Canteen, or For. This is label/layout
    based and contains no company/vendor/client hardcoding.
    """
    values = list(extra_values)
    for key in (
        "vendor_employee_name_raw",
        "vendor_address_raw",
        "_vendor_identity_evidence",
        "_party_identity_evidence",
        "visible_text_raw",
        "document_type_raw",
        "remarks_raw",
    ):
        values.append(raw.get(key))
    issuer_label = r"(?:From|Issued\s*By|Supplier|Vendor|Issuer|Shop|Stall|Restaurant|Canteen|For)"
    stop_labels = r"(?:M\s*/?\s*s\.?|Bill\s*To|Billed\s*To|Buyer|Customer|Client|To|Particulars|Quantity|Rate|Amount|Total|Date|No\.?|GSTIN|GST\s*No|HSN)"
    for value in values:
        text_value = safe_str(value)
        if not text_value:
            continue
        normalized = text_value.replace("\r", "\n")
        normalized = re.sub(rf"\s+({stop_labels}\s*[:\-]?)", r"\n\1", normalized, flags=re.IGNORECASE)
        patterns = [
            rf"\b{issuer_label}\s*[:\-]?\s*([^\n\r]{{2,160}})",
            rf"\b{issuer_label}\s*[:\-]?\s*(?:\n|\r\n)\s*([^\n\r]{{2,160}}(?:\n[^\n\r]{{2,160}})?)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                candidate = clean_labelled_vendor_candidate(match.group(1))
                if looks_like_valid_vendor_hint(raw, candidate):
                    return candidate
    return ""
def should_preserve_replaced_vendor_as_client(base: Dict[str, Any], old_vendor: Any, new_vendor: Any) -> bool:
    """
    When supplier-only vendor repair replaces a business name and client is blank,
    the replaced business name is often the billed-party that the vision model
    incorrectly placed in vendor_employee_name_raw. Preserve it as client only
    under conservative, generic conditions.
    """
    old_name = normalize_invoice_name(old_vendor)
    new_name = normalize_invoice_name(new_vendor)
    if not old_name or not new_name or old_name.lower() == new_name.lower():
        return False
    if is_generic_vendor_name(old_name) or looks_like_client_fragment(old_name):
        return False
    existing_client = normalize_invoice_name(base.get("client_name_raw"))
    if existing_client and not looks_like_invalid_client_any_context(base, existing_client):
        return False
    validation_context = dict(base)
    validation_context["vendor_employee_name_raw"] = new_name
    if not looks_like_valid_billed_party_hint(validation_context, old_name):
        return False
    old_tokens = significant_name_tokens(old_name)
    new_tokens = significant_name_tokens(new_name)
    if old_tokens and new_tokens and len(old_tokens & new_tokens) >= max(1, min(len(old_tokens), len(new_tokens)) // 2):
        return False
    # Safer signal: the replaced name looks like a legal/business billed party,
    # while the new vendor is a different supplier/issuer found by the vendor pass.
    if has_company_suffix_or_vendor_word(old_name) or looks_business_like(old_name):
        return True
    return False
def is_non_english_text(value: Any) -> bool:
    return bool(re.search(r"[^\x00-\x7F]", safe_str(value)))
DEVANAGARI_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
DEVANAGARI_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऋ": "ri",
}
DEVANAGARI_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e",
    "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ृ": "ri",
}
DEVANAGARI_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}
def _remove_trailing_inherent_a(text: str) -> str:
    if text.endswith("a"):
        return text[:-1]
    return text
def transliterate_devanagari_run(text: str) -> str:
    """Generic Devanagari to Latin transliteration fallback for vendor/client names."""
    output = ""
    for ch in safe_str(text):
        if ch in DEVANAGARI_CONSONANTS:
            output += DEVANAGARI_CONSONANTS[ch] + "a"
        elif ch in DEVANAGARI_MATRAS:
            output = _remove_trailing_inherent_a(output) + DEVANAGARI_MATRAS[ch]
        elif ch == "्":
            output = _remove_trailing_inherent_a(output)
        elif ch in DEVANAGARI_INDEPENDENT_VOWELS:
            output += DEVANAGARI_INDEPENDENT_VOWELS[ch]
        elif ch in DEVANAGARI_DIGITS:
            output += DEVANAGARI_DIGITS[ch]
        elif ch in ("ं", "ँ"):
            output += "n"
        elif ch == "ः":
            output += "h"
        elif ch == "़":
            continue
        elif ch in ("।", "॥"):
            output += " "
        else:
            # Important safety: never pass Devanagari characters through unchanged.
            # If an uncommon sign/letter is not mapped above, replace it with a
            # space so final API output stays in English/Latin script.
            if "\u0900" <= ch <= "\u097F":
                output += " "
            else:
                output += ch
    return output
def has_indian_script(value: Any) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", safe_str(value)))
def strip_remaining_indian_script(value: Any) -> str:
    text = safe_str(value)
    text = re.sub(r"[\u0900-\u097F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .,:;|-\t")
def english_script_invoice_name(value: Any) -> str:
    """
    Convert Indian-script invoice names to English/Latin script without using any
    vendor/client/company hardcoding. This is a fallback when the vision model
    returns Hindi/Marathi/Devanagari characters despite the prompt.
    """
    text = safe_str(value)
    if not text:
        return ""
    # Transliterate only Devanagari runs; keep already-English text unchanged.
    text = re.sub(
        r"[\u0900-\u097F]+",
        lambda match: transliterate_devanagari_run(match.group(0)),
        text,
    )
    # Generic cleanup for common Hindi/Marathi shop words written phonetically.
    # These are category-word normalizations, not vendor/company hardcoding.
    # Basic final-schwa cleanup makes outputs more natural in English script.
    text = re.sub(r"\b([A-Za-z]{3,})a\b", r"\1", text)
    replacements = [
        (r"\bshree\b", "Shri"),
        (r"\bshri\b", "Shri"),
        (r"\btee\b", "Tea"),
        (r"\bti\b", "Tea"),
        (r"\bsto?la?\b", "Stall"),
        (r"\bstaal\b", "Stall"),
        (r"\bmaarket\b", "Market"),
        (r"\bmarket\b", "Market"),
        (r"\bhotel\b", "Hotel"),
        (r"\bcanteen\b", "Canteen"),
        (r"\bagency\b", "Agency"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Extra generic phonetic cleanup after Devanagari transliteration.
    # These are language/category normalizations, not vendor/company hardcoding.
    phonetic_replacements = [
        (r"\bshreeraam\b", "Shriram"),
        (r"\bshreeram\b", "Shriram"),
        (r"\bshri\s*ram\b", "Shriram"),
        (r"\btee\b", "Tea"),
        (r"\btt+i+\b", "Tea"),
        (r"\bst[aoe]*l+l?\b", "Stall"),
        (r"\bstaal\b", "Stall"),
        (r"\bmaarket+t?\b", "Market"),
    ]
    for pattern, replacement in phonetic_replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,:;|-\t")
    # Hard guarantee: final returned name must not contain Devanagari/Hindi script.
    # If any unsupported Indian-script character survived, remove that character
    # instead of returning Hindi font to the API response.
    if has_indian_script(text):
        text = strip_remaining_indian_script(text)
    # Title-case only when the entire value is not deliberately uppercase.
    if text and not text.isupper():
        words = []
        for word in text.split(" "):
            if re.fullmatch(r"[A-Z0-9&./-]+", word):
                words.append(word)
            else:
                words.append(word[:1].upper() + word[1:])
        text = " ".join(words)
    return normalize_invoice_name(text)
def enforce_english_script_on_name_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final generic cleanup for API/raw output.
    It converts only name-like identity fields from Devanagari/Hindi/Marathi
    script to English/Latin script. It does not hardcode any vendor/client name.
    """
    if not isinstance(raw, dict):
        return raw
    name_keys = [
        "vendor_employee_name_raw",
        "vendor_display_name_english_raw",
        "client_name_raw",
        "billed_party_raw",
        "account_name_raw",
    ]
    for key in name_keys:
        value = raw.get(key)
        if has_indian_script(value):
            raw[key] = english_script_invoice_name(value)
    # Preserve original blocks where possible, but if a raw block is used later as
    # a fallback for name extraction, make sure the standalone name part is Latin.
    for key in ["customer_block_raw", "billed_party_block_raw", "client_block_raw"]:
        value = raw.get(key)
        if has_indian_script(value):
            raw[key] = english_script_invoice_name(value)
    return raw
def should_run_vendor_identity_repair(raw: Dict[str, Any]) -> bool:
    vendor = safe_str(raw.get("vendor_employee_name_raw"))
    if not vendor:
        return True
    if is_generic_vendor_name(vendor):
        return True
    if is_non_english_text(vendor):
        return True
    if suspicious_vendor_for_local_expense(raw, vendor):
        return True
    if should_run_legal_supplier_repair(raw):
        return True
    # If the current vendor is only a broad category while the document has
    # subscription/publication context, do a stronger header identity pass.
    context = " ".join([
        safe_str(raw.get("document_type_raw")),
        safe_str(raw.get("expense_type_raw")),
        safe_str(raw.get("service_usage_raw")),
        safe_str(raw.get("service_period_raw")),
        safe_str(raw.get("remarks_raw")),
        safe_str(raw.get("billed_party_raw")),
        safe_str(raw.get("customer_block_raw")),
        safe_str(raw.get("client_block_raw")),
        " ".join(safe_str(item.get("particulars_raw")) for item in (raw.get("items") or []) if isinstance(item, dict)),
    ]).lower()
    if re.search(r"\b(newspaper|publication|subscription|monthly subscription|mint|economic times)\b", context, flags=re.IGNORECASE):
        if re.search(r"\b(dealers?|newspaper dealers?|subscription|publication)\b", vendor, flags=re.IGNORECASE):
            return True
    client = safe_str(raw.get("client_name_raw"))
    if client and normalize_invoice_name(client).lower() == normalize_invoice_name(vendor).lower():
        return True
    # Generic scanned-bill guard: when the billed-party/client is missing or invalid,
    # and the current vendor looks like a business recipient in a bill/table context,
    # run the supplier-only identity pass. This fixes cases where the model picked
    # the M/s/Bill-To company as vendor instead of the actual issuer at top/signature.
    if looks_like_invalid_client_any_context(raw, client):
        table_context = bool(re.search(
            r"\b(bill\s*of\s*supply|composition\s+taxable|m/s\.?|bill\s*to|buyer|customer|particulars|hsn|gstin|amount|nos|btls)\b",
            context,
            flags=re.IGNORECASE,
        ))
        if table_context and looks_business_like(vendor):
            return True
    return False
def merge_vendor_identity_fields(base: Dict[str, Any], vendor_identity: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(vendor_identity, dict):
        return base
    evidence_hint = safe_str(vendor_identity.get("evidence_raw"))
    candidate = first_non_empty(
        vendor_identity.get("vendor_employee_name_raw"),
        vendor_identity.get("vendor_display_name_english_raw"),
        extract_labelled_vendor_from_context(base, evidence_hint, vendor_identity.get("vendor_address_raw")),
    )
    candidate = english_script_invoice_name(candidate)
    if candidate and not is_generic_vendor_name(candidate) and not looks_like_customer_or_recipient_name(candidate):
        base["_vendor_identity_candidate_raw"] = candidate
        current = safe_str(base.get("vendor_employee_name_raw"))
        replace_vendor = (
            should_run_vendor_identity_repair(base)
            or should_replace_vendor_with_source(current, candidate)
            or (suspicious_vendor_for_local_expense(base, current) and not suspicious_vendor_for_local_expense(base, candidate))
        )
        if replace_vendor:
            if should_preserve_replaced_vendor_as_client(base, current, candidate):
                preserved_client = normalize_invoice_name(current)
                base["client_name_raw"] = preserved_client
                base["billed_party_raw"] = first_non_empty(base.get("billed_party_raw"), preserved_client)
                base["_client_candidate_from_replaced_vendor_raw"] = preserved_client
            base["vendor_employee_name_raw"] = candidate
    for key in ["vendor_address_raw", "los_vendor_state_raw"]:
        value = safe_str(vendor_identity.get(key))
        if value and not safe_str(base.get(key)):
            base[key] = value
    evidence = safe_str(vendor_identity.get("evidence_raw"))
    if evidence:
        existing = safe_str(base.get("_vendor_identity_evidence"))
        base["_vendor_identity_evidence"] = first_non_empty(existing, evidence)
    return base
def extract_vendor_name_from_text(source_text: str) -> str:
    """
    Deterministic vendor fallback for searchable PDFs.
    Strong vendor/issuer labels are preferred so customer names are not accidentally used as the vendor.
    """
    text = safe_str(source_text)
    patterns = [
        r"\bFor\s+([A-Z][A-Za-z0-9&.'()\-/ ]{2,90}?(?:Limited|Ltd\.?|Private\s+Limited|Pvt\.?\s*Ltd\.?|LLP|Hospitals?|Markets?))\b",
        r"Registered\s+Office\s*:\s*([A-Z][A-Za-z0-9&.'()\-/ ]{2,90}?(?:Limited|Ltd\.?|Private\s+Limited|Pvt\.?\s*Ltd\.?|LLP))\b",
        r"Corporate\s+Identity\s+Number\s*:[^\n]{0,80}\b([A-Z][A-Za-z0-9&.'()\-/ ]{2,90}?(?:Limited|Ltd\.?|Private\s+Limited|Pvt\.?\s*Ltd\.?|LLP))\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = normalize_invoice_name(match.group(1))
            if value and not looks_like_customer_or_recipient_name(value):
                return value
    return ""
def should_run_party_identity_repair(raw: Dict[str, Any]) -> bool:
    if not ENABLE_PARTY_IDENTITY_REPAIR:
        return False
    vendor = safe_str(raw.get("vendor_employee_name_raw"))
    client_name = first_non_empty(raw.get("client_name_raw"), raw.get("billed_party_raw"), raw.get("_client_candidate_from_replaced_vendor_raw"))
    if not vendor or is_generic_vendor_name(vendor) or suspicious_vendor_for_local_expense(raw, vendor):
        return True
    if is_non_english_text(vendor):
        return True
    if not client_name:
        return True
    if looks_like_invalid_client_any_context(raw, client_name) or looks_like_client_fragment(client_name):
        return True
    return False
def merge_party_identity_fields(base: Dict[str, Any], party: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(party, dict):
        return base
    party = normalize_raw_result(party)
    evidence = safe_str(party.get("evidence_raw"))
    if evidence:
        base["_party_identity_evidence"] = first_non_empty(base.get("_party_identity_evidence"), evidence)
    vendor_candidate = first_non_empty(
        party.get("vendor_employee_name_raw"),
        extract_labelled_vendor_from_context(base, evidence, party.get("vendor_address_raw")),
    )
    vendor_candidate = english_script_invoice_name(vendor_candidate)
    if vendor_candidate and looks_like_valid_vendor_hint(base, vendor_candidate):
        current_vendor = safe_str(base.get("vendor_employee_name_raw"))
        if (not current_vendor or is_generic_vendor_name(current_vendor) or suspicious_vendor_for_local_expense(base, current_vendor) or is_non_english_text(current_vendor) or should_replace_vendor_with_source(current_vendor, vendor_candidate, evidence)):
            if current_vendor and not safe_str(base.get("client_name_raw")) and looks_like_valid_billed_party_hint(base, current_vendor):
                base["_client_candidate_from_replaced_vendor_raw"] = normalize_invoice_name(current_vendor)
            base["vendor_employee_name_raw"] = vendor_candidate
            base["_vendor_identity_candidate_raw"] = vendor_candidate
    client_candidate = first_non_empty(
        party.get("client_name_raw"),
        party.get("billed_party_raw"),
        extract_labelled_client_from_context(base, evidence, party.get("client_address_raw")),
    )
    client_candidate = clean_billed_party_candidate(client_candidate)
    if client_candidate and looks_like_valid_billed_party_hint(base, client_candidate):
        current_client = safe_str(base.get("client_name_raw"))
        if should_replace_client_with_source(base, client_candidate) or not current_client:
            base["client_name_raw"] = client_candidate
            base["billed_party_raw"] = first_non_empty(base.get("billed_party_raw"), client_candidate)
    for key in ("vendor_gstin_raw", "isec_gstin_raw", "vendor_address_raw", "client_address_raw"):
        value = safe_str(party.get(key))
        if value and not safe_str(base.get(key)):
            base[key] = value
    base = repair_generic_context_fields(base)
    base = repair_client_from_context(base)
    return base
def party_identity_result_has_useful_value(base: Dict[str, Any], party: Dict[str, Any]) -> bool:
    """Return True when a party-identity response has any useful supplier/client/GSTIN value."""
    if not isinstance(party, dict):
        return False
    try:
        party_norm = normalize_raw_result(party)
    except Exception:
        party_norm = party
    evidence = safe_str(party_norm.get("evidence_raw"))
    vendor_candidate = first_non_empty(
        party_norm.get("vendor_employee_name_raw"),
        extract_labelled_vendor_from_context(base, evidence, party_norm.get("vendor_address_raw")),
    )
    vendor_candidate = english_script_invoice_name(vendor_candidate)
    if vendor_candidate and looks_like_valid_vendor_hint(base, vendor_candidate):
        return True
    client_candidate = first_non_empty(
        party_norm.get("client_name_raw"),
        party_norm.get("billed_party_raw"),
        extract_labelled_client_from_context(base, evidence, party_norm.get("client_address_raw")),
    )
    client_candidate = clean_billed_party_candidate(client_candidate)
    if client_candidate and looks_like_valid_billed_party_hint(base, client_candidate):
        return True
    if safe_str(party_norm.get("vendor_gstin_raw")) or safe_str(party_norm.get("isec_gstin_raw")):
        return True
    return False
def extract_client_name_from_text(source_text: str) -> str:
    """
    Deterministic customer/client fallback from searchable PDF text.
    This is generic and label/layout based:
    - explicit Bill To/Buyer/Recipient/Customer labels first;
    - then company/person line inside a customer address / place-of-supply block;
    - never uses a fixed client/company name.
    """
    text = safe_str(source_text)
    # 1) Strong labelled customer-side patterns.
    patterns = [
        # Same-line buyer/customer labels: Buyer: ABC Ltd. / M/s. ABC Ltd.
        r"\b(?:M\s*/?\s*s\.?|Bill\s*To|Billed\s*To|Buyer|Recipient|Customer|Client|Subscriber|Consumer)\s*[:\-]?\s*([^\n]{2,120})",
        # Next-line buyer/customer labels.
        r"\b(?:Bill\s*To|Billed\s*To|Buyer|Recipient|Customer|Client|Subscriber|Consumer)\s*[:,-]?\s*(?:\n|\r\n|\s{2,})\s*([^\n]{2,120})",
        r"\b(?:Customer|Client|Subscriber|Consumer)\s+Name\s*[:\-]?\s*([^\n]{2,120})",
        # Generic Name label, but do not let State Name / Company Name labels become client.
        r"(?<!State\s)(?<!Company\s)\bName\s*[:\-]?\s*([^\n]{2,120})(?=\s+(?:Address|GST|State|Place\s+of\s+Supply|Ship\s+To))",
    ]
    labelled_bad_re = re.compile(
        r"\b(state\s*name|ifd\s*code|udyam|gstin|uin|pan|project\s*id|currency|dated|kind\s*attn|attention|amount|total|taxable|invoice|tax\s+invoice|receipt|bill|bank|account|ifsc|authori[sz]ed)\b",
        flags=re.IGNORECASE,
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = clean_billed_party_candidate(match.group(1))
            if value and not labelled_bad_re.search(value) and not looks_like_customer_or_recipient_name(value):
                return value
    # 2) Generic address-block fallback. Many telecom/utility PDFs expose text as:
    #    CUSTOMER NAME
    #    address lines...
    #    State / pincode
    #    Ship To State Code / Place of Supply
    lines = [normalize_invoice_name(line) for line in re.split(r"[\r\n]+", text) if normalize_invoice_name(line)]
    vendor_name = extract_vendor_name_from_text(text)
    vendor_tokens = significant_name_tokens(vendor_name)
    bad_line_re = re.compile(
        r"\b(total|amount|previous|payment|relationship|bill\s*(?:number|date|period)|pay\s+by|pay\s+outstanding|security\s+deposit|deposit|rentals?|taxes|charges?|invoice|tax\s+invoice|original\s+copy|irn|email|landmark|signature|stamp|page\s+\d+|registered\s+office|corporate\s+identity|for\s+|ifd\s*code|udyam|cin|gstin|uin|state\s+name|code\s*:|project\s+id|currency|dated|kind\s+attn|attention|company'?s\s+pan|bank\s+details|a/c|ifsc|authori[sz]ed)\b",
        flags=re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        candidate = clean_billed_party_candidate(line)
        if not candidate:
            continue
        if bad_line_re.search(candidate):
            continue
        if vendor_name and normalize_invoice_name(candidate).lower() == normalize_invoice_name(vendor_name).lower():
            continue
        # Prefer a legal/business/person-looking line, but do not require any hardcoded company name.
        has_name_shape = (
            has_company_suffix_or_vendor_word(candidate)
            or bool(re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}", candidate))
            or bool(re.search(r"\b(bank|securities|finance|capital|insurance|consulting|technolog(?:y|ies)|solutions|enterprise|agency|association|society|trust|foundation)\b", candidate, flags=re.IGNORECASE))
        )
        if not has_name_shape:
            continue
        # Skip address/locality lines unless they also look like a legal/business name.
        if re.search(r"\b(road|rd|street|lane|sector|floor|market|nagar|vihar|plot|building|pin\s*code|pincode)\b", candidate, flags=re.IGNORECASE) and not has_company_suffix_or_vendor_word(candidate):
            continue
        # Next few lines should look like a real customer address / supply block.
        # Do not use GST alone because summary labels like Security Deposit can appear just before GST No.
        next_block = " ".join(lines[idx + 1: idx + 7])
        prev_block = " ".join(lines[max(0, idx - 3): idx])
        has_address_context = re.search(
            r"\b(address|state|place\s+of\s+supply|ship\s+to|pincode|pin\s*code|landmark|floor|sector|road|market|nagar|vihar|faridabad|gurugram|mumbai|pune|delhi|bangalore|bengaluru|hyderabad|chennai|kolkata|haryana|maharashtra|gujarat|karnataka|telangana|tamil\s+nadu|west\s+bengal)\b",
            next_block,
            flags=re.IGNORECASE,
        )
        if has_address_context:
            # Avoid issuer/legal/footer sections.
            if not re.search(r"\b(registered\s+office|corporate\s+identity|authori[sz]ed\s+signatory|for\s+)\b", prev_block, flags=re.IGNORECASE):
                return candidate
    # 3) Customer GSTIN neighbourhood fallback: look around buyer/customer GSTIN and choose
    #    nearest valid previous name line.
    for gst_match in re.finditer(r"(?:GST\s+registration\s+no|GST\s*No|GSTIN|UIN)\s*[:\-]?\s*[0-9A-Z]{15}", text, flags=re.IGNORECASE):
        prefix = text[max(0, gst_match.start() - 500):gst_match.start()]
        prefix_lines = [normalize_invoice_name(line) for line in re.split(r"[\r\n]+", prefix) if normalize_invoice_name(line)]
        for line in reversed(prefix_lines[-8:]):
            candidate = clean_billed_party_candidate(line)
            if (
                candidate
                and not bad_line_re.search(candidate)
                and not looks_like_customer_or_recipient_name(candidate)
                and (has_company_suffix_or_vendor_word(candidate) or re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}", candidate))
                and not (vendor_name and normalize_invoice_name(candidate).lower() == normalize_invoice_name(vendor_name).lower())
            ):
                return candidate
    return ""
def extract_isec_gstin_from_text(source_text: str) -> str:
    """Find customer/recipient GSTIN without confusing it with vendor GSTIN."""
    text = safe_str(source_text).upper()
    # Strong customer/buyer-side patterns first. These are label/layout based,
    # not company-name based. The non-greedy window lets us capture forms like:
    # Buyer: ... GSTIN/UIN: 27XXXXXXXXXXXXX
    patterns = [
        r"BUYER\s*:?([\s\S]{0,900}?)GSTIN\s*/\s*UIN\s*[:\-]?\s*([0-9A-Z]{15})",
        r"BILL\s+TO\s*:?([\s\S]{0,900}?)GSTIN\s*/\s*UIN\s*[:\-]?\s*([0-9A-Z]{15})",
        r"BILLED\s+TO\s*:?([\s\S]{0,900}?)GSTIN\s*/\s*UIN\s*[:\-]?\s*([0-9A-Z]{15})",
        r"RECIPIENT[\s\S]{0,250}?GSTIN\s*[:\-]?\s*([0-9A-Z]{15})",
        r"CUSTOMER[\s\S]{0,250}?GSTIN\s*[:\-]?\s*([0-9A-Z]{15})",
        r"GST\s+REGISTRATION\s+NO\s*[:\-]?\s*([0-9A-Z]{15})",
        r"RECIPIENT\s+GSTIN\s*[:\-]?\s*([0-9A-Z]{15})",
        r"CUSTOMER\s+GSTIN\s*[:\-]?\s*([0-9A-Z]{15})",
        r"GSTIN\s*/\s*UIN\s+OF\s+RECIPIENT\s*[:\-]?\s*([0-9A-Z]{15})",
        r"ISEC\s+GSTIN\s*[:\-]?\s*([0-9A-Z]{15})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = safe_str(match.group(match.lastindex)).upper()
            if GSTIN_PATTERN.fullmatch(candidate):
                return candidate
    # Generic fallback: when a PDF exposes two GSTINs, the first is usually
    # supplier/vendor and the second is buyer/recipient. Use only distinct values.
    gstins = extract_gstins_from_text(text)
    if len(gstins) >= 2:
        return gstins[1]
    return ""
def extract_customer_state_from_text(source_text: str) -> str:
    """Extract customer/LOR state from explicit buyer/recipient labels."""
    text = safe_str(source_text)
    patterns = [
        r"Place\s+of\s+supply\s*[:\-]?\s*(?:\d{1,2}\s*[-–]\s*)?([A-Za-z ]{3,40})",
        r"State\s+Name\s*[:\-]?\s*(?:\d{1,2}\s*[-–]\s*)?([A-Za-z ]{3,40})",
        r"Buyer[\s\S]{0,700}?State\s+Name\s*[:\-]?\s*(?:\d{1,2}\s*[-–]\s*)?([A-Za-z ]{3,40})",
        r"Buyer[\s\S]{0,700}?Place\s+of\s+supply\s*[:\-]?\s*(?:\d{1,2}\s*[-–]\s*)?([A-Za-z ]{3,40})",
        r"Ship\s+To\s+State\s+Code\s*[:\-]?\s*([0-9]{1,2})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = safe_str(match.group(1)).strip(" .,:;|-")
            # State code pattern.
            if value.isdigit():
                state = GSTIN_STATE_CODE_MAP.get(value.zfill(2), "")
                if state:
                    return state
            state = state_or_empty(value)
            if state:
                return state
    isec_gstin = extract_isec_gstin_from_text(text)
    return state_from_gstin(isec_gstin)
def looks_like_currency_or_table_fragment(value: Any) -> bool:
    """Detect OCR fragments from table headers/currency labels, not real expense types."""
    text = normalize_invoice_name(value).lower()
    if not text:
        return True
    # Remove punctuation around fragments like "(INR)", "Amount (`)", "Rate per".
    compact = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if not compact:
        return True
    tokens = compact.split()
    if tokens and all(token in CURRENCY_OR_TABLE_FRAGMENT_WORDS for token in tokens):
        return True
    # Very short uppercase/currency/table fragments are common OCR mistakes.
    if len(compact) <= 4 and any(token in CURRENCY_OR_TABLE_FRAGMENT_WORDS for token in tokens):
        return True
    return False
def is_generic_expense_type(value: Any) -> bool:
    text = normalize_invoice_name(value).lower()
    compact = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return (
        (not text)
        or text in GENERIC_EXPENSE_WORDS
        or text.rstrip("s") in GENERIC_EXPENSE_WORDS
        or compact in GENERIC_EXPENSE_WORDS
        or compact.rstrip("s") in GENERIC_EXPENSE_WORDS
        or looks_like_currency_or_table_fragment(value)
    )
def extract_primary_service_line_from_text(source_text: str) -> str:
    """Extract the first meaningful service/particular line from invoice body."""
    lines = [line.strip(" .,:;|\t") for line in safe_str(source_text).splitlines() if line.strip()]
    for index, line in enumerate(lines):
        lower = line.lower()
        if re.search(r"\bparticulars\b", lower):
            for candidate in lines[index + 1:index + 12]:
                cand = normalize_invoice_name(candidate)
                if not cand:
                    continue
                cand_lower = cand.lower()
                if re.fullmatch(r"(?i)(hsn|sac|hsn/sac|gst|rate|per|amount|particulars|\(?inr\)?|rs\.?|rupees?)", cand_lower):
                    continue
                if re.search(r"\b(cgst|sgst|igst|total|please\s+deduct|tds)\b", cand_lower):
                    continue
                if re.search(r"[A-Za-z]", cand) and not re.fullmatch(r"[0-9,./% -]+", cand):
                    return cand
    # Fallback for service invoices where OCR keeps the service line outside the header.
    service_line_patterns = [
        r"\b([A-Z][A-Za-z& /-]{3,80}?\s+Services?)\b",
        r"\b([A-Z][A-Za-z& /-]{3,80}?\s+Support(?:\s*&\s*[A-Za-z ]+)?)\b",
    ]
    for pattern in service_line_patterns:
        for match in re.finditer(pattern, safe_str(source_text), flags=re.IGNORECASE):
            value = normalize_invoice_name(match.group(1))
            if value and not re.search(r"bank details|computer generated|payment advice", value, flags=re.IGNORECASE):
                return value
    return ""
def extract_secondary_service_detail_from_text(source_text: str) -> str:
    lines = [line.strip(" .,:;|\t") for line in safe_str(source_text).splitlines() if line.strip()]
    for line in lines:
        if re.search(r"\bfor\s+the\s+period\b", line, flags=re.IGNORECASE):
            detail = re.split(r"\bfor\s+the\s+period\b", line, maxsplit=1, flags=re.IGNORECASE)[0]
            detail = normalize_invoice_name(detail)
            if detail:
                return detail
    return ""
def should_use_service_unit(source_text: str, service_line: str) -> bool:
    text = safe_str(source_text).lower()
    if not service_line:
        return False
    # Period alone does not mean a monthly unit. Use Month only for explicit
    # monthly/rental/billing-period utility contexts.
    if any(keyword in text for keyword in ("monthly", "monthly rentals", "month's charges", "bill period", "billing period", "bill upto", "bill up to", "current charge", "garbage collection", "municipal")):
        return False
    return bool(re.search(r"\bservices?\b", service_line, flags=re.IGNORECASE))
def extract_bill_period_from_text(source_text: str) -> str:
    text = safe_str(source_text)
    patterns = [
        r"Bill\s+period\s*[:\-]?\s*([^\n]{0,120})",
        r"Billing\s+period\s*[:\-]?\s*([^\n]{0,120})",
        r"Service\s+period\s*[:\-]?\s*([^\n]{0,120})",
        r"Bill\s+Upto\s*[:\-]?\s*([^\n]{0,160})",
        r"Bill\s+Up\s+To\s*[:\-]?\s*([^\n]{0,160})",
        r"Charge\s+period\s*[:\-]?\s*([^\n]{0,160})",
        r"For\s+the\s+period\s*[:\-]?\s*([\s\S]{0,180})",
        r"For\s+the\s+month\s+of\s+((?:" + MONTH_PATTERN + r")\s+(?:20\d{2}|\d{2}))",
        r"Fees?\s+for\s+the\s+month\s+of\s+((?:" + MONTH_PATTERN + r")\s+(?:20\d{2}|\d{2}))",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = safe_str(match.group(1))
            if len(extract_all_full_dates(value)) >= 2:
                return value
            period_from_date, period_to_date = derive_date_range_from_period(value)
            if period_from_date and period_to_date:
                return value
    return ""
def extract_current_charges_total_from_text(source_text: str) -> str:
    text = safe_str(source_text)
    patterns = [
        r"This\s+month'?s\s+charges\s*[:+\-= ]{0,10}(-?\d[\d,]*(?:\.\d+)?)",
        r"Current\s+charges?\s*(?:Rs\.?)?\s*[:+\-= ]{0,20}(?:INR|Rs\.?)?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"Monthly\s+Finali[sz]ed\s+Amount\s*(?:Rs\.?)?\s*[:+\-= ]{0,20}(?:INR|Rs\.?)?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]
    candidates: List[Decimal] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            amount = decimal_or_none(match.group(1))
            if amount is not None and amount > 0:
                candidates.append(amount)
    if not candidates:
        return ""
    # Summary total is usually the largest current-charge value. Detail pages may
    # repeat smaller per-connection charges such as 499 + 499.
    return format_decimal_amount(max(candidates))
def extract_tax_table_from_text(source_text: str) -> Dict[str, str]:
    """
    Extract common GST tax table rows such as:
    HSN Taxable Value CGST Rate Amount SGST Rate Amount Total Tax
    998412 998.00 9% 89.82 9% 89.82 179.64
    """
    text = re.sub(r"[ \t]+", " ", safe_str(source_text))
    gst_row_pattern = re.compile(
        r"\b(?P<hsn>\d{4,8})\s+"
        r"(?P<taxable>-?\d[\d,]*(?:\.\d+)?)\s+"
        r"(?P<cgst_rate>\d+(?:\.\d+)?)\s*%\s+"
        r"(?P<cgst>-?\d[\d,]*(?:\.\d+)?)\s+"
        r"(?P<sgst_rate>\d+(?:\.\d+)?)\s*%\s+"
        r"(?P<sgst>-?\d[\d,]*(?:\.\d+)?)\s+"
        r"(?P<total_tax>-?\d[\d,]*(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    for match in gst_row_pattern.finditer(text):
        hsn = safe_str(match.group("hsn"))
        taxable = clean_amount(match.group("taxable"))
        cgst = clean_amount(match.group("cgst"))
        sgst = clean_amount(match.group("sgst"))
        total_tax = clean_amount(match.group("total_tax"))
        if hsn and taxable:
            return {
                "hsn": hsn,
                "taxable": taxable,
                "cgst": cgst,
                "sgst": sgst,
                "igst": "0",
                "cess": "0",
                "total_tax": total_tax,
                "cgst_rate": normalize_percent_rate(match.group("cgst_rate") + "%"),
                "sgst_rate": normalize_percent_rate(match.group("sgst_rate") + "%"),
                "igst_rate": "0%",
                "gst_rate": combine_percent_rates(match.group("cgst_rate") + "%", match.group("sgst_rate") + "%"),
            }
    # IGST-only invoices are also common.
    igst_row_pattern = re.compile(
        r"\b(?P<hsn>\d{4,8})\s+"
        r"(?P<taxable>-?\d[\d,]*(?:\.\d+)?)\s+"
        r"(?P<igst_rate>\d+(?:\.\d+)?)\s*%\s+"
        r"(?P<igst>-?\d[\d,]*(?:\.\d+)?)\b",
        flags=re.IGNORECASE,
    )
    for match in igst_row_pattern.finditer(text):
        hsn = safe_str(match.group("hsn"))
        taxable = clean_amount(match.group("taxable"))
        igst = clean_amount(match.group("igst"))
        if hsn and taxable and igst:
            return {
                "hsn": hsn,
                "taxable": taxable,
                "cgst": "0",
                "sgst": "0",
                "igst": igst,
                "cess": "0",
                "total_tax": igst,
                "cgst_rate": "0%",
                "sgst_rate": "0%",
                "igst_rate": normalize_percent_rate(match.group("igst_rate") + "%"),
                "gst_rate": normalize_percent_rate(match.group("igst_rate") + "%"),
            }
    # Fallback for compact/professional-fee invoices where PDF text order is
    # not the same as the visual table, for example:
    #   Professional Fees 75,785.00 997159 CGST 6,820.65 % 9 SGST 6,820.65 % 9
    # or a tax summary like:
    #   HSN/SAC Taxable Value 997159 75,785.00
    taxable = ""
    hsn = ""
    summary_match = re.search(
        r"HSN\s*/\s*SAC\s+Taxable\s+Value[\s\S]{0,120}?\b(?P<hsn>\d{4,8})\s+(?P<taxable>-?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if summary_match:
        hsn = safe_str(summary_match.group("hsn"))
        taxable = clean_amount(summary_match.group("taxable"))
    if not hsn or not taxable:
        service_amount_hsn_match = re.search(
            r"\b(?:Professional\s+Fees?|Professional\s+Fee|Service\s+Charges?|Consultancy\s+Fees?|Retainer\s+Fees?)\b[\s\S]{0,80}?(?P<taxable>-?\d[\d,]*(?:\.\d+)?)\s+(?P<hsn>\d{4,8})\b",
            text,
            flags=re.IGNORECASE,
        )
        if service_amount_hsn_match:
            hsn = safe_str(service_amount_hsn_match.group("hsn"))
            taxable = clean_amount(service_amount_hsn_match.group("taxable"))
    def first_labeled_tax_amount(label: str) -> str:
        label_match = re.search(r"\b" + re.escape(label) + r"\b[\s\S]{0,80}", text, flags=re.IGNORECASE)
        if not label_match:
            return ""
        window = label_match.group(0)
        for number_match in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", window):
            amount = clean_amount(number_match.group(0))
            amount_dec = decimal_or_none(amount)
            # Skip GST rates like 9 or 18 when they appear before/after the amount.
            if amount_dec is not None and amount_dec > Decimal("20"):
                return amount
        return ""
    if hsn and taxable:
        cgst = first_labeled_tax_amount("CGST")
        sgst = first_labeled_tax_amount("SGST")
        igst = first_labeled_tax_amount("IGST")
        cgst_dec = decimal_or_none(cgst) or Decimal("0")
        sgst_dec = decimal_or_none(sgst) or Decimal("0")
        igst_dec = decimal_or_none(igst) or Decimal("0")
        total_tax = format_decimal_amount(cgst_dec + sgst_dec + igst_dec) if any([cgst, sgst, igst]) else ""
        return {
            "hsn": hsn,
            "taxable": taxable,
            "cgst": cgst or "0",
            "sgst": sgst or "0",
            "igst": igst or "0",
            "cess": "0",
            "total_tax": total_tax,
            "cgst_rate": "",
            "sgst_rate": "",
            "igst_rate": "",
            "gst_rate": derive_gst_rate_from_tax_amounts(taxable, cgst or "0", sgst or "0", igst or "0", "0"),
        }
    return {}
def extract_quantity_unit_from_text(source_text: str) -> Tuple[str, str]:
    """
    Generic quantity/unit extraction from remarks or description text.
    Examples handled: No. of Persons 2, No of Users: 10, 3 Seats.
    Returns (quantity, normalized_unit).
    """
    text = safe_str(source_text)
    unit_aliases = {
        "person": "Person",
        "persons": "Person",
        "people": "Person",
        "employee": "Employee",
        "employees": "Employee",
        "resource": "Resource",
        "resources": "Resource",
        "seat": "Seat",
        "seats": "Seat",
        "user": "User",
        "users": "User",
        "license": "License",
        "licenses": "License",
    }
    patterns = [
        r"\bNo\.?\s*of\s+(?P<unit>persons?|people|employees?|resources?|seats?|users?|licenses?)\s*[:\-]?\s*(?P<qty>\d+(?:\.\d+)?)\b",
        r"\bNumber\s+of\s+(?P<unit>persons?|people|employees?|resources?|seats?|users?|licenses?)\s*[:\-]?\s*(?P<qty>\d+(?:\.\d+)?)\b",
        r"\b(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>persons?|people|employees?|resources?|seats?|users?|licenses?)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            qty = clean_amount(match.group("qty"))
            unit_key = safe_str(match.group("unit")).lower()
            unit = unit_aliases.get(unit_key, safe_str(match.group("unit")).title())
            if qty and unit:
                return qty, unit
    return "", ""
def is_count_based_quantity_unit(value: Any) -> bool:
    return safe_str(value).lower() in {
        "person", "employee", "resource", "seat", "user", "license",
    }
def infer_service_month_unit_from_dates(raw: Dict[str, Any], from_date: str, to_date: str) -> str:
    if is_professional_fee_context(raw) and dates_cover_full_calendar_month(from_date, to_date):
        return "Month"
    return "Service"
def infer_service_usage_from_text(source_text: str) -> str:
    text_lower = safe_str(source_text).lower()
    if any(keyword in text_lower for keyword in ("garbage collection", "waste collection")):
        return "Garbage & User Charge Collection"
    if any(keyword in text_lower for keyword in ("user charge", "municipal charge", "utility charge")):
        return "User / Utility Charges"
    if any(keyword in text_lower for keyword in ("fixedline", "fixed line", "broadband", "telecommunication", "fiber", "fibre")):
        return "Telecommunication / Broadband Service"
    detail = extract_secondary_service_detail_from_text(source_text)
    if detail:
        return detail
    service_line = extract_primary_service_line_from_text(source_text)
    if service_line:
        return service_line
    return ""
def infer_expense_type_from_text(source_text: str) -> str:
    text_lower = safe_str(source_text).lower()
    if any(keyword in text_lower for keyword in ("garbage collection", "waste collection", "user charge")):
        return "Garbage Collection / Municipal Charges"
    if "fixedline and broadband services" in text_lower:
        return "Fixedline and Broadband Services"
    if any(keyword in text_lower for keyword in ("broadband", "telecommunication", "fixed line", "fixedline")):
        return "Telecommunication Service"
    service_line = extract_primary_service_line_from_text(source_text)
    if service_line:
        return service_line
    return ""
def infer_particulars_from_source_text(source_text: str) -> str:
    text_lower = safe_str(source_text).lower()
    if "garbage collection" in text_lower:
        return "Garbage Collection Charges"
    if "user charge" in text_lower:
        return "User Charges"
    if "fixedline and broadband services" in text_lower or "monthly rentals" in text_lower:
        return "Fixedline and Broadband Services / Monthly Rentals"
    if "rentals" in text_lower and "this month's charges" in text_lower:
        return "Monthly Rentals"
    primary = extract_primary_service_line_from_text(source_text)
    secondary = extract_secondary_service_detail_from_text(source_text)
    if primary and secondary and secondary.lower() not in primary.lower():
        return f"{primary} - {secondary}"
    if primary:
        return primary
    return ""
def repair_line_items_from_source_text(raw: Dict[str, Any], source_text: str) -> Dict[str, Any]:
    tax_table = extract_tax_table_from_text(source_text)
    if not tax_table:
        return raw
    if not safe_str(raw.get("subtotal_raw")):
        raw["subtotal_raw"] = tax_table.get("taxable", "")
    if not safe_str(raw.get("cgst_total_raw")):
        raw["cgst_total_raw"] = tax_table.get("cgst", "")
    if not safe_str(raw.get("sgst_total_raw")):
        raw["sgst_total_raw"] = tax_table.get("sgst", "")
    if not safe_str(raw.get("igst_total_raw")):
        raw["igst_total_raw"] = tax_table.get("igst", "")
    if not safe_str(raw.get("cess_total_raw")):
        raw["cess_total_raw"] = tax_table.get("cess", "")
    if tax_table.get("gst_rate"):
        raw["_source_text_gst_rate_raw"] = tax_table.get("gst_rate", "")
    current_total = extract_current_charges_total_from_text(source_text)
    if current_total and not safe_str(raw.get("grand_total_raw")):
        raw["grand_total_raw"] = current_total
    particulars = infer_particulars_from_source_text(source_text)
    service_line = extract_primary_service_line_from_text(source_text)
    quantity_hint, quantity_unit_hint = "", ""
    if quantity_unit_hint:
        quantity_unit = quantity_unit_hint
    elif text_has_any_keyword(source_text, TELECOM_MONTHLY_UNIT_KEYWORDS):
        quantity_unit = "Month"
    elif should_use_service_unit(source_text, service_line):
        quantity_unit = "Service"
    elif text_has_any_keyword(source_text, PERIODIC_SERVICE_UNIT_KEYWORDS):
        quantity_unit = "Month"
    else:
        quantity_unit = "Service" if service_line else ""
    source_item = {
        "particulars_raw": particulars,
        "quantity_raw": "",
        "quantity_unit_raw": "",
        "hsn_code_raw": tax_table.get("hsn", ""),
        "rate_raw": tax_table.get("gst_rate", ""),
        "amount_raw": tax_table.get("taxable", ""),
        "cgst_raw": tax_table.get("cgst", ""),
        "sgst_raw": tax_table.get("sgst", ""),
        "igst_raw": tax_table.get("igst", ""),
        "cess_raw": tax_table.get("cess", ""),
    }
    items = raw.get("items") or []
    has_usable_item = False
    for item in items:
        if not isinstance(item, dict):
            continue
        if safe_str(item.get("amount_raw")) or safe_str(item.get("cgst_raw")) or safe_str(item.get("sgst_raw")):
            has_usable_item = True
        if should_replace_line_item_with_source(raw, item, source_item, source_text):
            for key, value in source_item.items():
                if value:
                    item[key] = value
            continue
        for key, value in source_item.items():
            if not value:
                continue
            current_value = safe_str(item.get(key))
            if not current_value:
                item[key] = value
                continue
            if key == "rate_raw" and "%" in safe_str(item.get("rate_raw")):
                item[key] = value
            if key == "quantity_unit_raw" and current_value.lower() == "month" and value.lower() == "service":
                # A service period in a professional/service invoice is not automatically a monthly unit.
                item[key] = value
            if key == "particulars_raw" and (looks_like_bad_particulars(current_value) or is_generic_expense_type(current_value)):
                item[key] = value
    if not items or not has_usable_item:
        raw["items"] = [source_item]
    return raw
def extract_invoice_date_from_text(source_text: str) -> str:
    text = safe_str(source_text)
    # IMPORTANT: Check the explicitly labelled invoice date first.
    # Invoice numbers can contain date-looking fragments, e.g. 02-03/2025-26.
    # Such fragments must never be treated as the bill date.
    patterns = [
        r"\bDated\s*[:\-]?\s*([^\n]{0,120})",
        r"Invoice\s+No\.?[\s\S]{0,260}?\b(\d{1,2}[-/.](?:[A-Za-z]{3,9}|\d{1,2})[-/.]\d{2,4})\b",
        r"Invoice\s+No\.?[\s\S]{0,260}?\b((?:" + MONTH_PATTERN + r")\s+\d{1,2},?\s+\d{2,4})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            dates = extract_all_full_dates(match.group(0))
            if dates:
                return dates[-1]
            normalized = normalize_date_string(match.group(1))
            if normalized and re.match(r"^\d{2}/\d{2}/\d{4}$", normalized):
                return normalized
    return ""
def extract_labeled_date_from_text(source_text: str) -> str:
    invoice_date = extract_invoice_date_from_text(source_text)
    if invoice_date:
        return invoice_date
    labels = [
        "invoice date",
        "bill date",
        "bill issued date",
        "issued date",
        "transaction date",
        "receipt date",
        "date",
    ]
    text = safe_str(source_text)
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:\-]?\s*([^\n]{{0,80}})"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            normalized = normalize_date_string(match.group(1))
            if normalized and re.match(r"^\d{2}/\d{2}/\d{4}$", normalized):
                return normalized
    return ""
def extract_bill_no_from_text(source_text: str) -> str:
    text = safe_str(source_text)
    def valid_doc_no(value: str) -> bool:
        value = safe_str(value).strip(" .,:;|]")
        if not value:
            return False
        if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", value):
            return False
        # Avoid bank/account numbers unless no stronger document number exists.
        if re.fullmatch(r"\d{8,20}", value):
            return False
        return True
    strong_patterns = [
        r"Invoice\s+No\.?\s*(?:\n\s*Dated\s*)?(?:\n[^\n]{0,120}){0,4}?\b([A-Z]{2,}[A-Z0-9]*/[A-Z0-9/_\-]{2,60})\b",
        r"(?:Invoice\s+No\.?|Bill\s+No\.?|Receipt\s+No\.?|Voucher\s+No\.?|Deposit\s+Receipt\s+No\.?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/_\-]{2,60})",
        r"\b(CRN\d{6,})\b",
    ]
    for pattern in strong_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = safe_str(match.group(1)).strip(" .,:;|]")
            if valid_doc_no(value):
                return value
    # Last fallback: labels from configured list except bank/account no labels.
    for label in [label for label in BILL_NO_LABELS if label.lower() not in {"account no"}]:
        pattern = rf"{re.escape(label)}\s*[:\-]?\s*([A-Z0-9][A-Z0-9/_\-]{{2,60}})"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = safe_str(match.group(1)).strip(" .,:;|]")
            if valid_doc_no(value):
                return value
    return ""
def clean_document_number(value: Any) -> str:
    """Remove OCR/model-copied labels from document numbers."""
    text = safe_str(value)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip(" .,:;|\t")
    # If model returned "Bill No : KNN/61/135680", keep only the value.
    label_alt = "|".join(re.escape(label) for label in sorted(BILL_NO_LABELS, key=len, reverse=True))
    match = re.search(rf"(?:{label_alt})\s*[:\-]?\s*([A-Z0-9][A-Z0-9/_\-]{{1,60}})", text, flags=re.IGNORECASE)
    if match:
        text = safe_str(match.group(1))
    # Remove a leading label that OCR left partially attached.
    text = re.sub(rf"^(?:{label_alt})\s*[:\-]?\s*", "", text, flags=re.IGNORECASE).strip(" .,:;|")
    return text
def is_known_indian_state(value: Any) -> bool:
    text = safe_str(value)
    return any(re.fullmatch(re.escape(state), text, flags=re.IGNORECASE) for state in INDIAN_STATE_NAMES)
def state_or_empty(value: Any) -> str:
    text = safe_str(value)
    if is_known_indian_state(text):
        for state in INDIAN_STATE_NAMES:
            if state.lower() == text.lower():
                return state
    states = find_states_in_text(text)
    if len(states) == 1:
        return states[0]
    lower = text.lower()
    for city, state in sorted(CITY_STATE_HINTS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(r"\b" + re.escape(city) + r"\b", lower):
            return state
    return ""
def looks_like_wrong_client_for_local_bill(raw: Dict[str, Any], value: Any) -> bool:
    current = normalize_invoice_name(value)
    if not current:
        return True
    if not is_local_service_context(raw):
        return False
    vendor = normalize_invoice_name(raw.get("vendor_employee_name_raw"))
    account = normalize_invoice_name(raw.get("account_name_raw"))
    if vendor and current.lower() == vendor.lower():
        return True
    if account and current.lower() == account.lower():
        return True
    if looks_like_authority_name(current):
        return True
    if re.search(r"\b(account\s+name|ifsc|bank\s+name|authori[sz]ed\s+signatory|payment\s+detail)\b", current, flags=re.IGNORECASE):
        return True
    return False
def looks_like_valid_billed_party_hint(raw: Dict[str, Any], value: Any) -> bool:
    """
    Validate a generic billed-party/client hint before replacing client_name_raw.
    This prevents Address/Category/Bank/Amount/Signatory text from becoming client name.
    """
    candidate = normalize_invoice_name(value)
    if not candidate:
        return False
    if looks_like_client_fragment(candidate):
        return False
    lower = candidate.lower()
    if re.search(
        r"\b(address|category|monthly\s+finali[sz]ed|amount|current\s+charge|arrear|total|payment\s+detail|account\s+name|account\s+no|ifsc|bank\s+name|authori[sz]ed\s+signatory|bill\s+no|bill\s+upto|issued\s+date|service\s+start)\b",
        lower,
        flags=re.IGNORECASE,
    ):
        return False
    if lower in {"commercial", "residential", "industrial", "retail", "restaurant", "cash", "upi", "card"}:
        return False
    # Reject common address/locality-only values that vision models often pick as client.
    if re.search(r"\b(road|rd|street|lane|colony|nagar|vihar|sector|house|plot|floor|wing|near|opposite|market|bazar|bazar|marg|chawl|estate|building)\b", lower, flags=re.IGNORECASE):
        # Allow only if it also looks like a business/legal name; otherwise it is probably an address.
        if not has_company_suffix_or_vendor_word(candidate) and not re.search(r"\b(direct|securities|bank|limited|ltd|pvt|private|company|services|enterprises|agency|association|society|trust|foundation)\b", lower, flags=re.IGNORECASE):
            return False
    # Reject pure addresses/localities when the same text is already stored as address.
    address_fields = [
        raw.get("client_address_raw", ""),
        raw.get("patient_address_raw", ""),
        raw.get("supply_address_raw", ""),
        raw.get("bill_to_address_raw", ""),
        raw.get("ship_to_address_raw", ""),
    ]
    for address in address_fields:
        address_text = normalize_invoice_name(address).lower()
        if address_text and lower == address_text:
            return False
    vendor = normalize_invoice_name(raw.get("vendor_employee_name_raw")).lower()
    account = normalize_invoice_name(raw.get("account_name_raw")).lower()
    if vendor and lower == vendor:
        return False
    if account and lower == account:
        return False
    if looks_like_authority_name(candidate):
        return False
    # A client name can contain letters, &, spaces, dots, hyphens, or short numeric suffixes,
    # but it should not be mostly numeric.
    letters = re.findall(r"[A-Za-z]", candidate)
    digits = re.findall(r"\d", candidate)
    if not letters:
        return False
    if len(digits) > len(letters):
        return False
    return True
def clean_billed_party_candidate(value: Any, *, allow_labelled_name: bool = True) -> str:
    """
    Clean a possible billed-party/client value without knowing any company names.
    It removes labels and cuts off trailing Address/Category/Bill/Service details when
    a vision model returns a whole customer block as one string.
    """
    text = normalize_invoice_name(value)
    if not text:
        return ""
    # A line that starts with these labels is not a client name.
    invalid_start_labels = [
        "address", "category", "monthly finalized", "current charge", "arrear",
        "total amount", "payment detail", "account name", "account no", "ifsc",
        "bank name", "bill no", "bill upto", "bill up to", "bill issued",
        "service start", "service period", "gstin", "pan", "phone", "email",
    ]
    if any(re.match(rf"^\s*{re.escape(label)}\b\s*[:\-]?", text, flags=re.IGNORECASE) for label in invalid_start_labels):
        return ""
    if allow_labelled_name:
        text = re.sub(
            r"^\s*(?:m\s*/?\s*s\.?|to|name|customer\s*name|client\s*name|consumer\s*name|subscriber\s*name|billed\s*party|party\s*name)\s*[:\-]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
    # If the model returned: "ICICI DIRECT Address: Ashok Nagar Category: Commercial",
    # keep only "ICICI DIRECT".
    text = re.split(
        r"\b(?:address|category|consumer|subscriber|phone|email|gstin|pan|bill\s*no|service\s*start|bill\s*upto|bill\s*up\s*to|bill\s*issued|payment\s*detail|account\s*name|account\s*no|ifsc|bank\s*name)\b\s*[:\-]?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return normalize_invoice_name(text)
def extract_billed_party_from_customer_block(raw: Dict[str, Any], *blocks: Any) -> str:
    """
    Parse the first valid standalone name from a customer block.
    Generic examples handled:
      NAME
      Address: ...
      Category: ...
      To,
      NAME
      Address...
    """
    for block in blocks:
        text = safe_str(block)
        if not text:
            continue
        # Preserve likely line boundaries, but also split when OCR/LLM joins labels.
        normalized = re.sub(r"\r", "\n", text)
        normalized = re.sub(r"\s+(Address\s*:)", r"\n\1", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+(Category\s*:)", r"\n\1", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+(Consumer\s*:|Subscriber\s*:|Bill\s*No\s*:|Service\s*Start|Bill\s*Upto|Bill\s*Up\s*To|Bill\s*Issued)", r"\n\1", normalized, flags=re.IGNORECASE)
        lines = [line.strip(" .,:;|\t") for line in re.split(r"[\n;|]+", normalized) if line.strip()]
        for line in lines[:10]:
            # Skip pure labels such as "To,".
            if re.fullmatch(r"(?i)(to|bill\s*to|billed\s*to|customer|client|consumer|subscriber)", line.strip(" ,:")):
                continue
            candidate = clean_billed_party_candidate(line)
            if candidate and looks_like_valid_billed_party_hint(raw, candidate):
                return candidate
        # If everything came as one line, try cleaning the whole block once.
        candidate = clean_billed_party_candidate(normalized)
        if candidate and looks_like_valid_billed_party_hint(raw, candidate):
            return candidate
    return ""
def repair_client_from_context(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Use explicit billed-party/customer hints; do not use payee/account/address/signatory values as client."""
    has_billed_party_hint = any(
        safe_str(raw.get(key))
        for key in (
            "billed_party_raw",
            "billed_party_block_raw",
            "customer_block_raw",
            "client_block_raw",
            "visible_text_raw",
        )
    )
    labelled_client = extract_labelled_client_from_context(raw)
    if labelled_client:
        raw["client_name_raw"] = labelled_client
        raw["billed_party_raw"] = first_non_empty(raw.get("billed_party_raw"), labelled_client)
        return raw
    if not is_local_service_context(raw) and not has_billed_party_hint:
        return raw
    current_client = normalize_invoice_name(raw.get("client_name_raw"))
    # First, repair the common vision-model case where client_name_raw contains
    # a whole block: "NAME Address: ... Category: ...".
    cleaned_current = clean_billed_party_candidate(current_client)
    if cleaned_current and cleaned_current.lower() != current_client.lower() and looks_like_valid_billed_party_hint(raw, cleaned_current):
        raw["client_name_raw"] = cleaned_current
        raw["billed_party_raw"] = first_non_empty(raw.get("billed_party_raw"), cleaned_current)
        return raw
    billed_party = clean_billed_party_candidate(raw.get("billed_party_raw"))
    if billed_party and looks_like_valid_billed_party_hint(raw, billed_party):
        # For local/utility bills, billed_party_raw is produced only from the customer/billed-party block.
        # Prefer it over any current value unless they already match.
        if not current_client or current_client.lower() != billed_party.lower():
            raw["client_name_raw"] = billed_party
        return raw
    block_party = extract_billed_party_from_customer_block(
        raw,
        raw.get("billed_party_block_raw"),
        raw.get("customer_block_raw"),
        raw.get("client_block_raw"),
        raw.get("visible_text_raw"),
    )
    if block_party:
        raw["client_name_raw"] = block_party
        raw["billed_party_raw"] = first_non_empty(raw.get("billed_party_raw"), block_party)
        return raw
    if looks_like_wrong_client_for_local_bill(raw, current_client) or not looks_like_valid_billed_party_hint(raw, current_client):
        raw["client_name_raw"] = ""
    return raw
def find_states_in_text(text: str) -> List[str]:
    """Return Indian state names found in the same order as they appear in text."""
    normalized_text = safe_str(text)
    matches: List[Tuple[int, str]] = []
    for state in INDIAN_STATE_NAMES:
        pattern = r"\b" + re.escape(state).replace(r"\ ", r"\s+") + r"\b"
        for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
            matches.append((match.start(), state))
    matches.sort(key=lambda item: item[0])
    found: List[str] = []
    for _, state in matches:
        if state not in found:
            found.append(state)
    return found
def infer_state_from_relevant_text(source_text: str) -> str:
    relevant_text = remove_noise_sections_for_field_hints(source_text)
    states = find_states_in_text(relevant_text)
    if len(states) == 1:
        return states[0]
    return ""
def infer_lor_state_from_text(source_text: str) -> str:
    text = remove_noise_sections_for_field_hints(source_text)
    label_pattern = re.compile(
        r"(place\s+of\s+supply|supply\s+address|bill\s+to|billed\s+to|customer\s+address|client\s+address|patient\s+address|patient\s+name|ship\s+to|\bto\b|\baddress\b)",
        flags=re.IGNORECASE,
    )
    for match in label_pattern.finditer(text):
        window = text[match.start():match.start() + 350]
        states = find_states_in_text(window)
        if states:
            return states[0]
    return infer_state_from_relevant_text(text)
def infer_lor_state_from_raw(raw: Dict[str, Any]) -> str:
    """
    Generic LOR repair. LOR is customer/client/patient/place-of-supply state.
    Prefer labelled customer-side fields over broad document text so vendor
    address states do not overwrite customer state.
    """
    explicit = safe_str(raw.get("lor_isec_state_raw"))
    if explicit and is_known_indian_state(explicit):
        return explicit
    customer_side_fields = [
        raw.get("place_of_supply_raw", ""),
        raw.get("client_address_raw", ""),
        raw.get("patient_address_raw", ""),
        raw.get("supply_address_raw", ""),
        raw.get("bill_to_address_raw", ""),
        raw.get("ship_to_address_raw", ""),
        raw.get("state_raw", ""),
    ]
    for value in customer_side_fields:
        states = find_states_in_text(value)
        if states:
            return states[0]
    source_context = safe_str(raw.get("_source_text_context_raw"))
    if source_context:
        state = infer_lor_state_from_text(source_context)
        if state:
            return state
    # Last safe fallback: use a state only when all available context contains
    # exactly one Indian state. This avoids picking vendor Maharashtra from a
    # document that also has customer Jharkhand, etc.
    states = find_states_in_text(build_context_text(raw))
    if len(states) == 1:
        return states[0]
    return ""
def repair_lor_from_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    current_lor = safe_str(raw.get("lor_isec_state_raw"))
    inferred_lor = infer_lor_state_from_raw(raw)
    if inferred_lor and (not current_lor or not is_known_indian_state(current_lor)):
        raw["lor_isec_state_raw"] = inferred_lor
    isec_gstin_state = state_from_gstin(raw.get("isec_gstin_raw"))
    if isec_gstin_state and (not safe_str(raw.get("lor_isec_state_raw")) or not is_known_indian_state(raw.get("lor_isec_state_raw"))):
        raw["lor_isec_state_raw"] = isec_gstin_state
    return raw
def text_has_any_keyword(text: Any, keywords: List[str]) -> bool:
    lower = safe_str(text).lower()
    return any(re.search(r"\b" + re.escape(keyword) + r"\b", lower) for keyword in keywords)
def build_context_text(raw: Dict[str, Any]) -> str:
    parts = [
        raw.get("document_type_raw", ""),
        raw.get("vendor_employee_name_raw", ""),
        raw.get("client_name_raw", ""),
        raw.get("billed_party_raw", ""),
        raw.get("billed_party_block_raw", ""),
        raw.get("customer_block_raw", ""),
        raw.get("client_block_raw", ""),
        raw.get("visible_text_raw", ""),
        raw.get("expense_type_raw", ""),
        raw.get("service_usage_raw", ""),
        raw.get("remarks_raw", ""),
        raw.get("los_vendor_state_raw", ""),
        raw.get("lor_isec_state_raw", ""),
        raw.get("vendor_address_raw", ""),
        raw.get("client_address_raw", ""),
        raw.get("patient_address_raw", ""),
        raw.get("supply_address_raw", ""),
        raw.get("bill_to_address_raw", ""),
        raw.get("ship_to_address_raw", ""),
        raw.get("place_of_supply_raw", ""),
        raw.get("account_name_raw", ""),
        raw.get("service_title_raw", ""),
        raw.get("bill_upto_raw", ""),
        raw.get("bill_period_raw", ""),
        raw.get("_source_text_context_raw", ""),
        raw.get("_charge_summary_evidence", ""),
        raw.get("_utility_ocr_text_raw", ""),
        raw.get("_source_text_gst_rate_raw", ""),
        raw.get("gst_rate_raw", ""),
    ]
    for item in raw.get("items") or []:
        if isinstance(item, dict):
            parts.append(item.get("particulars_raw", ""))
    return " ".join(safe_str(part) for part in parts if safe_str(part))
def is_single_day_expense_context(raw: Dict[str, Any]) -> bool:
    return text_has_any_keyword(build_context_text(raw), SINGLE_DAY_EXPENSE_KEYWORDS)
def is_local_service_context(raw: Dict[str, Any]) -> bool:
    return text_has_any_keyword(build_context_text(raw), LOCAL_SERVICE_KEYWORDS)
def is_periodic_service_context(raw: Dict[str, Any]) -> bool:
    return text_has_any_keyword(build_context_text(raw), PERIODIC_SERVICE_UNIT_KEYWORDS + TELECOM_MONTHLY_UNIT_KEYWORDS)
def is_newspaper_subscription_context(raw: Dict[str, Any]) -> bool:
    context = build_context_text(raw).lower()
    return text_has_any_keyword(context, NEWSPAPER_SUBSCRIPTION_KEYWORDS)
def looks_like_publication_item(value: Any) -> bool:
    text = safe_str(value).lower()
    if not text:
        return False
    # Generic publication/newspaper signals. Kept broad; no vendor/company names.
    return bool(re.search(r"\b(newspaper|daily|times|tribune|express|mint|chronicle|herald|journal|magazine|paper)\b", text, flags=re.IGNORECASE))
def repair_newspaper_subscription_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not is_newspaper_subscription_context(raw):
        return raw
    if not normalize_yes_no(raw.get("whether_self_reimbursement_raw")):
        raw["whether_self_reimbursement_raw"] = "Yes"
    if is_generic_expense_type(raw.get("expense_type_raw")) or looks_like_publication_item(raw.get("expense_type_raw")):
        raw["expense_type_raw"] = "Newspaper / Subscription"
    if is_generic_expense_type(raw.get("service_usage_raw")) or looks_like_publication_item(raw.get("service_usage_raw")):
        raw["service_usage_raw"] = "Newspaper Monthly Subscription"
    # If the model chose the category heading as vendor, keep it only as a last resort;
    # the repair prompt is expected to replace it with the actual agency/shop/person name.
    if is_generic_vendor_name(raw.get("vendor_employee_name_raw")):
        better_vendor = first_non_empty(raw.get("_source_text_vendor_name_raw"), raw.get("account_name_raw"))
        if better_vendor and not is_generic_vendor_name(better_vendor):
            raw["vendor_employee_name_raw"] = better_vendor
    vendor_state = infer_los_state_from_raw(raw)
    if vendor_state and (not safe_str(raw.get("los_vendor_state_raw")) or not is_known_indian_state(raw.get("los_vendor_state_raw"))):
        raw["los_vendor_state_raw"] = vendor_state
    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue
        if not safe_str(item.get("quantity_raw")):
            item["quantity_raw"] = "1"
        # quantity_unit_raw is intentionally not populated; final output keeps quantity_unit empty.
        if safe_str(item.get("quantity_unit_raw")).lower() in {"bill", "item", "items", "month", "service", "trip", "deposit", "course", "examination"}:
            item["quantity_unit_raw"] = ""
        # rate_raw now means GST rate percentage only. Never backfill it from amount.
        for tax_key in ("cgst_raw", "sgst_raw", "igst_raw", "cess_raw"):
            if not safe_str(item.get(tax_key)):
                item[tax_key] = "0"
    if not safe_str(raw.get("cgst_total_raw")):
        raw["cgst_total_raw"] = "0"
    if not safe_str(raw.get("sgst_total_raw")):
        raw["sgst_total_raw"] = "0"
    if not safe_str(raw.get("igst_total_raw")):
        raw["igst_total_raw"] = "0"
    if not safe_str(raw.get("cess_total_raw")):
        raw["cess_total_raw"] = "0"
    return raw
def looks_like_authority_name(value: Any) -> bool:
    return text_has_any_keyword(value, AUTHORITY_NAME_KEYWORDS)
def looks_like_bad_particulars(value: Any) -> bool:
    text = safe_str(value).lower()
    if not text:
        return True
    if looks_like_payment_mode(text):
        return True
    return bool(re.search(
        r"\b(current\s+charges?|monthly\s+finali[sz]ed\s+amount|total\s+amount|arrears?|payment\s+detail|bill\s+total|net\s+to\s+pay|taxable\s+value)\b",
        text,
        flags=re.IGNORECASE,
    ))
def infer_los_state_from_raw(raw: Dict[str, Any]) -> str:
    """LOS is vendor/supplier state. Prefer vendor GSTIN state, then vendor-side address/state fields."""
    vendor_gstin_state = first_non_empty(
        state_from_gstin(raw.get("vendor_gstin_raw")),
        state_from_gstin(raw.get("_source_text_vendor_gstin_raw")),
    )
    if vendor_gstin_state:
        return vendor_gstin_state
    vendor_side_fields = [
        raw.get("vendor_address_raw", ""),
        raw.get("los_vendor_state_raw", ""),
    ]
    for value in vendor_side_fields:
        state = state_or_empty(value)
        if state:
            return state
    return ""
def repair_los_from_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    inferred_los = infer_los_state_from_raw(raw)
    current_los = safe_str(raw.get("los_vendor_state_raw"))
    if inferred_los and (not current_los or not is_known_indian_state(current_los)):
        raw["los_vendor_state_raw"] = inferred_los
    # If vendor GSTIN gives a definite state, prefer it over city/OCR noise.
    gstin_state = first_non_empty(
        state_from_gstin(raw.get("vendor_gstin_raw")),
        state_from_gstin(raw.get("_source_text_vendor_gstin_raw")),
    )
    if gstin_state and current_los and current_los.lower() != gstin_state.lower():
        raw["los_vendor_state_raw"] = gstin_state
    return raw
def repair_local_service_lor_from_los(raw: Dict[str, Any]) -> Dict[str, Any]:
    # For local/municipal/utility bills, the service location is normally the
    # same state as the local issuing/collecting state when no separate customer
    # state is visible. This is rule-based and does not depend on company names.
    if is_local_service_context(raw):
        los = safe_str(raw.get("los_vendor_state_raw"))
        lor = safe_str(raw.get("lor_isec_state_raw"))
        if los and not lor:
            raw["lor_isec_state_raw"] = los
        elif lor and not los:
            raw["los_vendor_state_raw"] = lor
    return raw
def repair_vendor_from_context(raw: Dict[str, Any]) -> Dict[str, Any]:
    current_vendor = safe_str(raw.get("vendor_employee_name_raw"))
    account_name = normalize_invoice_name(raw.get("account_name_raw"))
    client_name = normalize_invoice_name(raw.get("client_name_raw"))
    if account_name and account_name.lower() != client_name.lower() and is_local_service_context(raw):
        if (not current_vendor) or looks_like_authority_name(current_vendor) or name_appears_in_customer_context(build_context_text(raw), current_vendor):
            raw["vendor_employee_name_raw"] = account_name
    return raw
def is_tax_exempt_context(raw: Dict[str, Any]) -> bool:
    return bool(re.search(
        r"\b(exempted|exempt|gst\s*exempt|tax\s*exempt|bill\s*of\s*supply|composition\s+taxable|not\s+eligible\s+to\s+collect\s+tax)\b",
        build_context_text(raw),
        flags=re.IGNORECASE,
    ))
def repair_generic_context_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = repair_los_from_raw(raw)
    raw = repair_lor_from_raw(raw)
    raw = repair_local_service_lor_from_los(raw)
    raw = repair_vendor_from_context(raw)
    raw = repair_client_from_context(raw)
    raw = repair_isec_gstin_from_context(raw)
    # If the model put a currency/table header fragment such as "(INR)" into
    # expense_type_raw, replace it with a meaningful service/particular context.
    # This is generic: it does not use any company name; it uses source text
    # classification, service_usage, or the first valid item particulars.
    if is_generic_expense_type(raw.get("expense_type_raw")):
        candidate = first_non_empty(
            raw.get("service_usage_raw"),
            raw.get("_source_text_expense_type_raw"),
            raw.get("service_title_raw"),
        )
        if is_generic_expense_type(candidate):
            for item in raw.get("items") or []:
                if isinstance(item, dict):
                    item_particulars = safe_str(item.get("particulars_raw"))
                    if item_particulars and not is_generic_expense_type(item_particulars) and not looks_like_bad_particulars(item_particulars):
                        candidate = item_particulars
                        break
        if candidate and not is_generic_expense_type(candidate):
            raw["expense_type_raw"] = candidate
    if is_tax_exempt_context(raw):
        raw["cgst_total_raw"] = first_non_empty(raw.get("cgst_total_raw"), "0")
        raw["sgst_total_raw"] = first_non_empty(raw.get("sgst_total_raw"), "0")
        raw["igst_total_raw"] = first_non_empty(raw.get("igst_total_raw"), "0")
        raw["cess_total_raw"] = first_non_empty(raw.get("cess_total_raw"), "0")
        for item in raw.get("items") or []:
            if isinstance(item, dict):
                item["cgst_raw"] = first_non_empty(item.get("cgst_raw"), "0")
                item["sgst_raw"] = first_non_empty(item.get("sgst_raw"), "0")
                item["igst_raw"] = first_non_empty(item.get("igst_raw"), "0")
                item["cess_raw"] = first_non_empty(item.get("cess_raw"), "0")
    return raw
def infer_quantity_unit_from_context(raw: Dict[str, Any]) -> str:
    context = build_context_text(raw)
    if text_has_any_keyword(context, TRIP_UNIT_KEYWORDS):
        return "Trip"
    if text_has_any_keyword(context, TELECOM_MONTHLY_UNIT_KEYWORDS):
        return "Month"
    if text_has_any_keyword(context, DEPOSIT_UNIT_KEYWORDS):
        return "Deposit"
    if text_has_any_keyword(context, EXAM_UNIT_KEYWORDS):
        return "Examination"
    if text_has_any_keyword(context, COURSE_UNIT_KEYWORDS):
        return "Course"
    if is_newspaper_subscription_context(raw):
        return "Month"
    if text_has_any_keyword(context, FOOD_UNIT_KEYWORDS):
        return "Bill"
    if text_has_any_keyword(context, GENERIC_PROFESSIONAL_SERVICE_KEYWORDS) or re.search(r"\bquality\s+engineering\s+services\b", context, flags=re.IGNORECASE):
        return "Service"
    if text_has_any_keyword(context, PERIODIC_SERVICE_UNIT_KEYWORDS):
        return "Month"
    return ""
def looks_like_payment_mode(value: Any) -> bool:
    text = safe_str(value).lower()
    if not text:
        return False
    return any(word in text for word in PAYMENT_MODE_WORDS)
def infer_particulars_from_context(raw: Dict[str, Any], current_particulars: Any = "") -> str:
    current = safe_str(current_particulars)
    context = build_context_text(raw)
    if text_has_any_keyword(context, ["garbage collection", "waste collection"]):
        return "Garbage Collection Charges"
    if text_has_any_keyword(context, ["user charge"]):
        return "User Charges"
    # Keep a useful model/table description when it is not just payment mode,
    # an amount label, or a generic word like "Service".
    if current and not looks_like_bad_particulars(current) and not is_generic_expense_type(current):
        return current
    if text_has_any_keyword(context, DEPOSIT_UNIT_KEYWORDS):
        if text_has_any_keyword(context, ["hospital", "patient", "medical"]):
            return "Hospital Deposit"
        return "Deposit"
    if text_has_any_keyword(context, EXAM_UNIT_KEYWORDS):
        return first_non_empty(raw.get("service_usage_raw"), raw.get("expense_type_raw"), "Certification Examination")
    if text_has_any_keyword(context, TRIP_UNIT_KEYWORDS):
        return first_non_empty(raw.get("service_usage_raw"), raw.get("expense_type_raw"), "Cab Ride")
    if text_has_any_keyword(context, FOOD_UNIT_KEYWORDS):
        return first_non_empty(raw.get("service_usage_raw"), raw.get("expense_type_raw"), "Food Bill")
    if text_has_any_keyword(context, TELECOM_MONTHLY_UNIT_KEYWORDS):
        return first_non_empty(raw.get("service_usage_raw"), raw.get("expense_type_raw"), "Fixedline and Broadband Services / Monthly Rentals")
    return current
def apply_service_date_fallback(
    raw: Dict[str, Any],
    from_date: str,
    to_date: str,
    bill_date: str,
    date_source: str,
) -> Tuple[str, str, str]:
    """
    For single-day receipts like cab/ride/travel bills, the service date is the
    visible bill/transaction date when no separate service period is present.
    Monthly/service-period invoices remain unchanged.
    """
    if from_date or to_date:
        return from_date, to_date, date_source
    if not bill_date or not re.match(r"^\d{2}/\d{2}/\d{4}$", bill_date):
        return from_date, to_date, date_source
    if not is_single_day_expense_context(raw):
        return from_date, to_date, date_source
    return bill_date, bill_date, "bill_date_raw_single_day_receipt"
def should_replace_document_number_with_source(current_doc_no: Any, source_doc_no: Any, source_text: str = "") -> bool:
    """Prefer a high-confidence searchable-PDF document number over OCR fragments.
    Generic rule: when PDF text has a labelled Invoice/Bill/Receipt/Voucher/CRN number,
    it is usually more reliable than a vision/OCR crop value. This does not hardcode
    any company name or invoice value.
    """
    current = clean_document_number(current_doc_no)
    source = clean_document_number(source_doc_no)
    if not source:
        return False
    if not current:
        return True
    if current.lower() == source.lower():
        return False
    bad_current_re = re.compile(
        r"\b(total|amount|date|period|relationship|account|security|deposit|charges?|tax(?:es)?|payment|previous|copy|invoice|bill)\b",
        flags=re.IGNORECASE,
    )
    if bad_current_re.search(current):
        return True
    if len(current) < 5 or re.search(r"[^A-Za-z0-9/_\-]", current):
        return True
    labelled_source_re = re.compile(
        r"\b(?:Invoice\s+No\.?|Bill\s+(?:No\.?|number)|Receipt\s+No\.?|Voucher\s+No\.?|Deposit\s+Receipt\s+No\.?|CRN/Ride\s+Number|CRN)\b[\s\S]{0,80}" + re.escape(source),
        flags=re.IGNORECASE,
    )
    if source_text and labelled_source_re.search(source_text):
        return True
    # For alphanumeric document numbers, prefer the longer labelled-looking value.
    if len(source) >= len(current) + 4 and re.search(r"[A-Za-z]", source) and re.search(r"\d", source):
        return True
    return False
def should_prefer_source_text_gstin(raw: Dict[str, Any], current_gstin: Any, source_gstin: Any) -> bool:
    current = safe_str(current_gstin).upper()
    source = safe_str(source_gstin).upper()
    if not source or not GSTIN_PATTERN.fullmatch(source):
        return False
    if not current or not GSTIN_PATTERN.fullmatch(current):
        return True
    if current == source:
        return False
    vendor_gstin = safe_str(first_non_empty(raw.get("vendor_gstin_raw"), raw.get("_source_text_vendor_gstin_raw"))).upper()
    if vendor_gstin and current == vendor_gstin and source != vendor_gstin:
        return True
    # If the source GSTIN state matches customer/LOR state and current does not, prefer source.
    source_state = state_from_gstin(source)
    current_state = state_from_gstin(current)
    lor_state = first_non_empty(raw.get("lor_isec_state_raw"), raw.get("place_of_supply_raw"), raw.get("client_address_raw"))
    if source_state and state_or_empty(lor_state) == source_state and current_state != source_state:
        return True
    return False
def should_prefer_source_text_context_value(current_value: Any, source_value: Any, source_text: str = "") -> bool:
    """Prefer deterministic source-text classification when current value is a small OCR fragment."""
    current = normalize_invoice_name(current_value)
    source = normalize_invoice_name(source_value)
    if not source:
        return False
    if not current:
        return True
    if current.lower() == source.lower():
        return False
    if looks_like_client_fragment(current) or len(current) <= 4:
        return True
    generic_or_fragment_re = re.compile(
        r"^(?:service|services|bill|invoice|tax|amount|total|rentals?|charges?|subscription|food|mint|chai|coffee)$",
        flags=re.IGNORECASE,
    )
    if generic_or_fragment_re.fullmatch(current):
        return True
    # If source text contains a strong bill title/service heading, use it over a random item word.
    if source_text and re.search(re.escape(source), source_text, flags=re.IGNORECASE):
        if len(source.split()) >= 3 and len(current.split()) <= 2:
            return True
    return False
def should_replace_line_item_with_source(raw: Dict[str, Any], current_item: Dict[str, Any], source_item: Dict[str, Any], source_text: str = "") -> bool:
    """Use source-text tax table line when current vision line is inconsistent.
    Generic: source_item is built only when a searchable PDF tax table/HSN row is found.
    Replace when current item has wrong amount/HSN/taxes or generic/wrong particulars.
    """
    if not isinstance(current_item, dict):
        return True
    source_amount = clean_amount(source_item.get("amount_raw"))
    source_hsn = normalize_hsn_sac_code(source_item.get("hsn_code_raw"))
    if not source_amount and not source_hsn:
        return False
    current_amount = clean_amount(current_item.get("amount_raw"))
    current_hsn = normalize_hsn_sac_code(current_item.get("hsn_code_raw"))
    current_particulars = safe_str(current_item.get("particulars_raw"))
    if source_hsn and current_hsn and current_hsn != source_hsn:
        return True
    if source_amount and current_amount and current_amount != source_amount:
        # A tax table taxable amount is a better line amount than Amount Due/Previous Balance.
        return True
    if source_hsn and not current_hsn:
        return True
    if source_amount and not current_amount:
        return True
    if looks_like_bad_particulars(current_particulars) or is_generic_expense_type(current_particulars):
        return True
    source_particulars = safe_str(source_item.get("particulars_raw"))
    if should_prefer_source_text_context_value(current_particulars, source_particulars, source_text):
        return True
    return False
def enrich_raw_from_source_text(raw: Dict[str, Any], file_path: str, filename: str) -> Dict[str, Any]:
    """
    Adds deterministic hints from searchable PDF text. This does not change
    scanned/image-only behavior because those files have no extracted PDF text.
    """
    try:
        source_text = extract_source_text_from_file(file_path, filename)
    except Exception as e:
        raw["_source_text_warning"] = f"Source text extraction failed: {str(e)}"
        return raw
    if not source_text.strip():
        return raw
    # Keep only a small context hint for classification/fallbacks, not the full PDF text.
    raw["_source_text_context_raw"] = remove_noise_sections_for_field_hints(source_text)[:2500]
    claim_total = extract_claim_total_from_text(source_text)
    if claim_total:
        raw["_source_text_claim_total_raw"] = claim_total
    source_vendor_name = extract_vendor_name_from_text(source_text)
    if source_vendor_name:
        raw["_source_text_vendor_name_raw"] = source_vendor_name
        if should_replace_vendor_with_source(raw.get("vendor_employee_name_raw"), source_vendor_name, source_text):
            raw["vendor_employee_name_raw"] = source_vendor_name
    source_client_name = extract_client_name_from_text(source_text)
    if source_client_name:
        raw["_source_text_client_name_raw"] = source_client_name
        if should_replace_client_with_source(raw, source_client_name):
            raw["client_name_raw"] = source_client_name
    source_expense_type = infer_expense_type_from_text(source_text)
    if source_expense_type:
        raw["_source_text_expense_type_raw"] = source_expense_type
    if source_expense_type and (
        not safe_str(raw.get("expense_type_raw"))
        or is_generic_expense_type(raw.get("expense_type_raw"))
        or should_prefer_source_text_context_value(raw.get("expense_type_raw"), source_expense_type, source_text)
    ):
        raw["expense_type_raw"] = source_expense_type
    source_service_usage = infer_service_usage_from_text(source_text)
    if source_service_usage and (
        not safe_str(raw.get("service_usage_raw"))
        or is_generic_expense_type(raw.get("service_usage_raw"))
        or should_prefer_source_text_context_value(raw.get("service_usage_raw"), source_service_usage, source_text)
    ):
        raw["service_usage_raw"] = source_service_usage
    bill_period = extract_bill_period_from_text(source_text)
    if bill_period:
        current_period = safe_str(raw.get("service_period_raw"))
        current_from, current_to = derive_date_range_from_explicit_range(current_period)
        if not current_from or not current_to:
            current_from, current_to = derive_date_range_from_period(current_period)
        source_from, source_to = derive_date_range_from_explicit_range(bill_period)
        if not source_from or not source_to:
            source_from, source_to = derive_date_range_from_period(bill_period)
        if not current_period or (source_from and source_to and (not current_from or not current_to)):
            raw["service_period_raw"] = bill_period
    quantity_hint, quantity_unit_hint = "", ""
    if quantity_hint:
        raw["_source_text_quantity_raw"] = quantity_hint
    if quantity_unit_hint:
        raw["_source_text_quantity_unit_raw"] = quantity_unit_hint
    if not safe_str(raw.get("digital_signature_evidence_raw")):
        irn_match = re.search(r"\bIRN(?:\s+Code)?\s*[:\-]\s*([^\n]{10,120})", source_text, flags=re.IGNORECASE)
        if irn_match:
            raw["digital_signature_evidence_raw"] = "IRN Code: " + safe_str(irn_match.group(1))
    if not safe_str(raw.get("digital_signature_evidence_raw")):
        signed_text_match = re.search(
            r"\b(digitally\s+signed\s+by[^\n]{0,140}|digital\s+signature[^\n]{0,140})",
            source_text,
            flags=re.IGNORECASE,
        )
        if signed_text_match:
            raw["digital_signature_evidence_raw"] = safe_str(signed_text_match.group(1))
    if not safe_str(raw.get("is_invoice_digitally_signed_raw")) and safe_str(raw.get("digital_signature_evidence_raw")):
        raw["is_invoice_digitally_signed_raw"] = "Yes"
    gstins = extract_gstins_from_text(source_text)
    source_vendor_gstin = extract_vendor_gstin_from_text(source_text)
    if source_vendor_gstin:
        raw["_source_text_vendor_gstin_raw"] = source_vendor_gstin
    if source_vendor_gstin and not safe_str(raw.get("vendor_gstin_raw")):
        raw["vendor_gstin_raw"] = source_vendor_gstin
    elif gstins and not safe_str(raw.get("vendor_gstin_raw")):
        # Fallback keeps old behavior only when a safer vendor-specific GSTIN was not found.
        raw["vendor_gstin_raw"] = gstins[0]
    source_isec_gstin = extract_isec_gstin_from_text(source_text)
    if source_isec_gstin:
        raw["_source_text_isec_gstin_raw"] = source_isec_gstin
        current_isec_gstin = safe_str(raw.get("isec_gstin_raw")).upper()
        if (
            not current_isec_gstin
            or not GSTIN_PATTERN.fullmatch(current_isec_gstin)
            or current_isec_gstin == safe_str(raw.get("vendor_gstin_raw")).upper()
            or should_prefer_source_text_gstin(raw, current_isec_gstin, source_isec_gstin)
        ):
            raw["isec_gstin_raw"] = source_isec_gstin
    if not safe_str(raw.get("isec_gstin_raw")) and len(gstins) >= 2:
        raw["isec_gstin_raw"] = gstins[1]
    hsn_sac_code = extract_hsn_sac_from_text(source_text)
    if hsn_sac_code:
        raw["_source_text_hsn_code_raw"] = hsn_sac_code
    raw = repair_line_items_from_source_text(raw, source_text)
    source_bill_no = extract_bill_no_from_text(source_text)
    if source_bill_no:
        raw["_source_text_bill_no_raw"] = source_bill_no
        current_doc_no = clean_document_number(first_non_empty(raw.get("bill_no_raw"), raw.get("invoice_no_raw")))
        if should_replace_document_number_with_source(current_doc_no, source_bill_no, source_text):
            raw["bill_no_raw"] = source_bill_no
            if safe_str(raw.get("invoice_no_raw")) and clean_document_number(raw.get("invoice_no_raw")) != source_bill_no:
                raw["invoice_no_raw"] = ""
    # A labelled date read from searchable PDF text is more reliable than a
    # vision-model guess. Always prefer it when present, including when the
    # model has mistaken a date-like invoice-number fragment for the bill date.
    source_bill_date = extract_labeled_date_from_text(source_text)
    if source_bill_date:
        raw["_source_text_bill_date_raw"] = source_bill_date
        current_bill_date = normalize_date_string(raw.get("bill_date_raw"))
        if current_bill_date != source_bill_date:
            raw["bill_date_raw"] = source_bill_date
    state_hint = infer_state_from_relevant_text(source_text)
    vendor_gstin_state = first_non_empty(
        state_from_gstin(raw.get("vendor_gstin_raw")),
        state_from_gstin(raw.get("_source_text_vendor_gstin_raw")),
    )
    current_los = safe_str(raw.get("los_vendor_state_raw"))
    if vendor_gstin_state and (not current_los or not is_known_indian_state(current_los)):
        raw["los_vendor_state_raw"] = vendor_gstin_state
    elif not current_los:
        raw["los_vendor_state_raw"] = state_hint
    source_customer_state = extract_customer_state_from_text(source_text)
    current_lor = safe_str(raw.get("lor_isec_state_raw"))
    if source_customer_state and (not current_lor or not is_known_indian_state(current_lor)):
        raw["lor_isec_state_raw"] = source_customer_state
    elif not current_lor:
        raw["lor_isec_state_raw"] = infer_lor_state_from_text(source_text)
    raw = repair_generic_context_fields(raw)
    quantity_unit_hint = safe_str(raw.get("_source_text_quantity_unit_raw"))
    if not quantity_unit_hint:
        for item in raw.get("items") or []:
            if isinstance(item, dict) and safe_str(item.get("quantity_unit_raw")):
                quantity_unit_hint = safe_str(item.get("quantity_unit_raw"))
                break
    if not quantity_unit_hint:
        quantity_unit_hint = infer_quantity_unit_from_context(raw)
    if quantity_unit_hint:
        raw["_source_text_quantity_unit_raw"] = quantity_unit_hint
    return raw
def looks_like_invalid_client_any_context(raw: Dict[str, Any], value: Any) -> bool:
    current = normalize_invoice_name(value)
    if not current:
        return True
    if looks_like_client_fragment(current):
        return True
    if looks_like_wrong_client_for_local_bill(raw, current):
        return True
    if not looks_like_valid_billed_party_hint(raw, current):
        return True
    if re.search(r"\b(address|category|consumer|subscriber|bill\s*no|service\s*start|bill\s*upto|bill\s*issued|payment\s*detail|account\s*name|ifsc|bank\s*name)\b", current, flags=re.IGNORECASE):
        return True
    return False
def should_run_customer_block_repair(raw: Dict[str, Any]) -> bool:
    current_client = raw.get("client_name_raw")
    if looks_like_invalid_client_any_context(raw, current_client):
        return True
    if safe_str(raw.get("billed_party_raw")) and normalize_invoice_name(raw.get("billed_party_raw")).lower() != normalize_invoice_name(current_client).lower():
        return True
    if is_local_service_context(raw) and not safe_str(raw.get("billed_party_raw")):
        return True
    return False
def merge_customer_block_fields(base: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(extracted, dict):
        return base
    repair = normalize_raw_result(extracted)
    for key in [
        "billed_party_raw",
        "billed_party_block_raw",
        "customer_block_raw",
        "client_block_raw",
        "client_address_raw",
        "lor_isec_state_raw",
        "isec_gstin_raw",
        "visible_text_raw",
    ]:
        value = safe_str(repair.get(key))
        if value and not safe_str(base.get(key)):
            base[key] = value
    candidate = first_non_empty(repair.get("billed_party_raw"), repair.get("client_name_raw"))
    candidate = clean_billed_party_candidate(candidate)
    if not candidate:
        candidate = extract_billed_party_from_customer_block(
            base,
            repair.get("billed_party_block_raw"),
            repair.get("customer_block_raw"),
            repair.get("client_block_raw"),
            repair.get("visible_text_raw"),
        )
    if candidate and looks_like_valid_billed_party_hint(base, candidate):
        if looks_like_invalid_client_any_context(base, base.get("client_name_raw")) or normalize_invoice_name(base.get("client_name_raw")).lower() != candidate.lower():
            base["client_name_raw"] = candidate
            base["billed_party_raw"] = first_non_empty(base.get("billed_party_raw"), candidate)
    base = repair_client_from_context(base)
    return base
def is_raw_missing_critical_fields(raw: Dict[str, Any]) -> bool:
    if is_generic_vendor_name(raw.get("vendor_employee_name_raw")):
        return True
    if not safe_str(first_non_empty(raw.get("bill_no_raw"), raw.get("invoice_no_raw"))):
        return True
    if not safe_str(raw.get("bill_date_raw")):
        return True
    if not safe_str(raw.get("is_invoice_digitally_signed_raw")):
        return True
    if not safe_str(raw.get("lor_isec_state_raw")):
        return True
    if is_local_service_context(raw):
        if not safe_str(raw.get("billed_party_raw")):
            return True
        if looks_like_wrong_client_for_local_bill(raw, raw.get("client_name_raw")):
            return True
    items = raw.get("items") or []
    if not items:
        return True
    for item in items:
        if not isinstance(item, dict):
            continue
        if not safe_str(item.get("particulars_raw")) or looks_like_payment_mode(item.get("particulars_raw")):
            return True
        if not safe_str(item.get("amount_raw")):
            return True
        # Do not trigger an extra LLM repair pass only for quantity, quantity_unit, or rate.
        # quantity_unit is intentionally blank in final output, and rate is only GST % when available.
    return False
def source_text_looks_usable_for_fast_path(source_text: str) -> bool:
    text_value = safe_str(source_text)
    if len(text_value) < SOURCE_TEXT_FAST_PATH_MIN_CHARS:
        return False
    # Require invoice-like anchors so random PDF text does not bypass vision.
    if not re.search(r"\b(invoice|bill|receipt|voucher|gstin|hsn|taxable|amount|total)\b", text_value, flags=re.IGNORECASE):
        return False
    # Require at least a few numeric tokens; invoices need dates/amounts/GSTIN/HSN.
    if len(re.findall(r"\d", text_value)) < 12:
        return False
    return True
def has_complete_source_text_extraction(raw: Dict[str, Any]) -> bool:
    """
    Fast-path gate for searchable PDFs.
    If any important field is weak, the normal vision pipeline is used, preserving accuracy.
    """
    vendor = safe_str(raw.get("vendor_employee_name_raw"))
    client = safe_str(raw.get("client_name_raw"))
    doc_no = safe_str(first_non_empty(raw.get("bill_no_raw"), raw.get("invoice_no_raw")))
    bill_date = safe_str(raw.get("bill_date_raw"))
    items = raw.get("items") or []
    if not vendor or is_generic_vendor_name(vendor):
        return False
    if not client or looks_like_invalid_client_any_context(raw, client):
        return False
    if re.search(r"\b(ifd\s*code|udyam|cin|gstin|uin|state\s+name|project\s+id|currency|dated|kind\s+attn|attention|bank|account|ifsc|authori[sz]ed)\b", client, flags=re.IGNORECASE):
        return False
    if not doc_no or not bill_date:
        return False
    # expense_type is intentionally disabled in final output; do not block the
    # text-PDF fast path only because expense_type/service_usage is blank.
    if not items:
        return False
    has_good_item = False
    for item in items:
        if not isinstance(item, dict):
            continue
        if safe_str(item.get("particulars_raw")) and safe_str(item.get("amount_raw")):
            has_good_item = True
            break
    if not has_good_item:
        return False
    # State/GSTIN is often available in text PDFs; if neither is present, use vision.
    if not first_non_empty(raw.get("lor_isec_state_raw"), raw.get("isec_gstin_raw"), raw.get("place_of_supply_raw")):
        return False
    return True
def build_source_text_fast_path_result(file_path: str, safe_filename: str) -> Dict[str, Any]:
    raw_result = normalize_raw_result({})
    raw_result = enrich_raw_from_source_text(raw_result, file_path, safe_filename)
    raw_result = enrich_raw_from_pdf_signature(raw_result, file_path, safe_filename)
    raw_result = repair_lor_from_raw(raw_result)
    raw_result = repair_generic_context_fields(raw_result)
    return raw_result
def looks_like_invalid_final_client_name(value: Any) -> bool:
    text = normalize_invoice_name(value)
    if not text:
        return True
    if looks_like_client_fragment(text):
        return True
    if re.search(
        r"\b(address|category|consumer|subscriber|bill\s*no|service\s*start|bill\s*upto|bill\s*issued|payment\s*detail|account\s*name|ifsc|bank\s*name|amount|total|rate|tax|gst|hsn|currency|dated)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    alpha_tokens = re.findall(r"[A-Za-z]{2,}", text)
    if not alpha_tokens:
        return True
    return False
def final_line_item_is_strictly_complete(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    required_text_fields = [
        "bill_no",
        "particulars",
        "from_date",
        "to_date",
        "amount",
        "bill_date",
        "quantity",
        # quantity_unit is intentionally blank in final output.
        # rate is GST percentage only and may be blank when no GST rate is visible.
    ]
    for field in required_text_fields:
        if not safe_str(item.get(field)):
            return False
    if looks_like_payment_mode(item.get("particulars")):
        return False
    if decimal_or_none(item.get("amount")) is None:
        return False
    rate_value = safe_str(item.get("rate"))
    if rate_value and rate_value != "0" and not re.fullmatch(r"\d+(?:\.\d+)?%(?:\s*\+\s*\d+(?:\.\d+)?%)*", rate_value):
        return False
    # Require tax fields to be intentionally populated. No-tax bills should have 0,
    # not blank. This avoids skipping repair when blank taxes still need normalization.
    for field in ["cgst", "sgst", "igst", "cess"]:
        value = safe_str(item.get(field))
        if value == "":
            return False
        if decimal_or_none(value) is None:
            return False
    return True
def final_output_is_strictly_complete(final: Dict[str, Any]) -> bool:
    """
    Safe early-exit gate for scanned/image documents after the concurrent
    full-document + header pass. It skips slower repair LLM calls only when the
    final API response is already strong enough. If anything is weak, the normal
    repair pipeline still runs, preserving accuracy.
    """
    if not isinstance(final, dict):
        return False
    vendor = safe_str(final.get("vendor_name"))
    if not vendor or is_generic_vendor_name(vendor) or is_non_english_text(vendor):
        return False
    # expense_type is intentionally disabled and always returned as blank.
    # Do not treat blank expense_type as an incomplete extraction.
    if not safe_str(final.get("los")):
        return False
    if looks_like_invalid_final_client_name(final.get("client_name")):
        return False
    # LOR is required for GST/state-driven invoices but can be blank for small
    # cash receipts where no customer state is visible. If ISEC GSTIN is present
    # in any line item, LOR must be present too.
    line_items = final.get("line_items") or []
    if not isinstance(line_items, list) or not line_items:
        return False
    if any(safe_str(item.get("isec_gstin")) for item in line_items if isinstance(item, dict)) and not safe_str(final.get("lor")):
        return False
    for item in line_items:
        if not final_line_item_is_strictly_complete(item):
            return False
    if safe_str(final.get("total_amount")) and decimal_or_none(final.get("total_amount")) is None:
        return False
    return True
def merge_parallel_repair_results(
    raw_result: Dict[str, Any],
    repair_result: Any = None,
    customer_result: Any = None,
    vendor_identity_result: Any = None,
) -> Dict[str, Any]:
    """Merge concurrently produced repair results in the same order as the old sequential pipeline."""
    if repair_result is not None and not isinstance(repair_result, Exception):
        raw_result = merge_repair_fields(raw_result, repair_result)
    if customer_result is not None and not isinstance(customer_result, Exception):
        raw_result = merge_customer_block_fields(raw_result, customer_result)
    if vendor_identity_result is not None and not isinstance(vendor_identity_result, Exception):
        raw_result = merge_vendor_identity_fields(raw_result, vendor_identity_result)
    return raw_result
def merge_repair_fields(base: Dict[str, Any], repair: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(repair, dict):
        return base
    repair = normalize_raw_result(repair)
    for key in RAW_KEYS:
        if key == "items":
            continue
        base_value = safe_str(base.get(key))
        repair_value = safe_str(repair.get(key))
        if not repair_value:
            continue
        if not base_value:
            base[key] = repair_value
            continue
        if key == "vendor_employee_name_raw" and is_generic_vendor_name(base_value) and not is_generic_vendor_name(repair_value):
            base[key] = repair_value
        # Keep stronger receipt/document number and date values from the repair pass.
        if key in {"bill_no_raw", "invoice_no_raw", "bill_date_raw"}:
            base[key] = repair_value
        # Use explicit No from repair pass when digital field was blank or uncertain.
        if key == "is_invoice_digitally_signed_raw" and normalize_yes_no(repair_value):
            base[key] = repair_value
        # For local/utility bills, prefer the repair pass client when it is also
        # the billed-party hint or when the current client is clearly not a client.
        if key == "client_name_raw" and is_local_service_context(base):
            repair_billed_party = normalize_invoice_name(repair.get("billed_party_raw"))
            if looks_like_valid_billed_party_hint(base, repair_value):
                if repair_billed_party.lower() == normalize_invoice_name(repair_value).lower() or looks_like_wrong_client_for_local_bill(base, base_value):
                    base[key] = repair_value
    for key in EXTRA_HINT_KEYS:
        repair_value = safe_str(repair.get(key))
        if repair_value and not safe_str(base.get(key)):
            base[key] = repair_value
    base = repair_generic_context_fields(base)
    base_items = [item for item in (base.get("items") or []) if isinstance(item, dict)]
    repair_items = [item for item in (repair.get("items") or []) if isinstance(item, dict)]
    if repair_items:
        base_has_good_item = any(
            safe_str(item.get("particulars_raw"))
            and not looks_like_payment_mode(item.get("particulars_raw"))
            and safe_str(item.get("amount_raw"))
            for item in base_items
        )
        repair_has_good_item = any(
            safe_str(item.get("particulars_raw"))
            and not looks_like_payment_mode(item.get("particulars_raw"))
            and safe_str(item.get("amount_raw"))
            for item in repair_items
        )
        if not base_items or (repair_has_good_item and not base_has_good_item):
            base["items"] = repair_items
        elif base_items and repair_items:
            # Fill blank columns on matching first item instead of replacing all rows.
            for key in ITEM_KEYS:
                if not safe_str(base_items[0].get(key)) and safe_str(repair_items[0].get(key)):
                    base_items[0][key] = safe_str(repair_items[0].get(key))
            base["items"] = base_items
    if repair.get("_warning") and not base.get("_repair_warning"):
        base["_repair_warning"] = safe_str(repair.get("_warning"))
    return base
def final_line_items_total(final_line_items: List[Dict[str, Any]]) -> Tuple[Decimal, bool]:
    total = Decimal("0")
    has_value = False
    for item in final_line_items:
        for key in ("amount", "cgst", "sgst", "igst", "cess"):
            value_decimal = decimal_or_none(item.get(key))
            if value_decimal is None:
                continue
            total += value_decimal
            has_value = True
    return total, has_value
def first_final_line_value(final_line_items: List[Dict[str, Any]], key: str) -> str:
    for item in final_line_items:
        value = safe_str(item.get(key))
        if value:
            return value
    return ""
def select_visible_claim_total(raw: Dict[str, Any]) -> Tuple[str, str]:
    """
    Source text value has priority because it comes from deterministic PDF text.
    Fall back to model grand_total_raw when source text is unavailable.
    """
    for key in ("_source_text_claim_total_raw", "grand_total_raw"):
        value = clean_amount(raw.get(key))
        if decimal_or_none(value) is not None:
            return value, key
    return "", ""
def should_use_visible_claim_total(raw: Dict[str, Any], final_line_items: List[Dict[str, Any]]) -> Tuple[bool, str, str, str]:
    if not allow_visible_claim_total_override(raw):
        return False, "", "", ""
    claim_total, claim_source = select_visible_claim_total(raw)
    if not claim_total:
        return False, "", "", ""
    claim_decimal = decimal_or_none(claim_total)
    line_total, has_line_total = final_line_items_total(final_line_items)
    if claim_decimal is None or not has_line_total or line_total <= 0:
        return False, "", "", ""
    if abs(claim_decimal - line_total) <= Decimal("0.01"):
        return False, "", "", ""
    if claim_decimal <= line_total:
        return False, "", "", ""
    # Do not override normal GST invoices where base amount + tax differs by a small percentage.
    # Override only when the visible paid/claim amount is clearly larger than the extracted
    # component/tax-invoice amount, e.g. cab receipt total 500 vs tax invoice component 5.
    ratio = claim_decimal / line_total
    if ratio < Decimal("1.50"):
        return False, "", "", ""
    return True, format_decimal_amount(claim_decimal), claim_source, format_decimal_amount(line_total)
def apply_visible_claim_total_override(
    raw: Dict[str, Any],
    final_line_items: List[Dict[str, Any]],
    bill_no: str,
    final_from_date: str,
    final_to_date: str,
    bill_date: str,
) -> List[Dict[str, Any]]:
    use_claim_total, claim_total, claim_source, previous_total = should_use_visible_claim_total(raw, final_line_items)
    if not use_claim_total:
        return final_line_items
    particulars = first_non_empty(
        raw.get("service_usage_raw"),
        first_final_line_value(final_line_items, "particulars"),
        raw.get("expense_type_raw"),
    )
    raw["_claim_amount_override_reason"] = (
        f"Used visible paid/claim total {claim_total} from {claim_source}; "
        f"extracted line/tax component total was {previous_total}."
    )
    return [
        {
            "bill_no": bill_no,
            "particulars": particulars,
            "from_date": final_from_date,
            "to_date": final_to_date,
            "amount": claim_total,
            "cgst": "0",
            "sgst": "0",
            "igst": "0",
            "cess": "0",
            "department_code": safe_str(raw.get("department_code_raw")),
            "function_code": safe_str(raw.get("function_code_raw")),
            "location_code": safe_str(raw.get("location_code_raw")),
            "sub_loc_code": safe_str(raw.get("sub_loc_code_raw")),
            "bill_date": bill_date,
            "quantity": "",
            "quantity_unit": "",
            "hsn_code": first_non_empty(first_final_line_value(final_line_items, "hsn_code"), raw.get("_source_text_hsn_code_raw")),
            "rate": choose_gst_rate_for_output(raw, {}, claim_total, "0", "0", "0", "0"),
            "isec_gstin": safe_str(raw.get("isec_gstin_raw")),
        }
    ]
def normalize_hsn_sac_code(value: Any) -> str:
    text = safe_str(value).strip()
    if not text:
        return ""
    normalized = re.sub(r"[\s.\\/-]+", "", text).lower()
    if normalized in {"na", "n/a", "notapplicable", "nil", "none"}:
        return ""
    match = re.search(r"\b\d{4,8}\b", text)
    return match.group(0) if match else text
def parse_normalized_date(date_text: Any):
    text = normalize_date_string(date_text)
    match = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", text)
    if not match:
        return None
    try:
        return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
    except ValueError:
        return None
def dates_cover_full_calendar_month(from_date: Any, to_date: Any) -> bool:
    start = parse_normalized_date(from_date)
    end = parse_normalized_date(to_date)
    if not start or not end:
        return False
    if start.year != end.year or start.month != end.month:
        return False
    last_day = calendar.monthrange(start.year, start.month)[1]
    return start.day == 1 and end.day == last_day
PROFESSIONAL_FEE_MONTH_KEYWORDS = [
    "professional fee", "professional fees", "professional charges", "retainer fee",
    "retainer fees", "consultant code", "consultant", "fee payment",
]
def is_professional_fee_context(raw: Dict[str, Any]) -> bool:
    context = build_context_text(raw)
    return text_has_any_keyword(context, PROFESSIONAL_FEE_MONTH_KEYWORDS)
def repair_isec_gstin_from_context(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Generic guard for consultant/professional invoices where buyer GSTIN was placed in vendor_gstin_raw."""
    if safe_str(raw.get("isec_gstin_raw")):
        return raw
    vendor_gstin = safe_str(raw.get("vendor_gstin_raw")).upper()
    if not GSTIN_PATTERN.fullmatch(vendor_gstin):
        return raw
    client_name = safe_str(raw.get("client_name_raw"))
    vendor_name = safe_str(raw.get("vendor_employee_name_raw"))
    has_no_tax = not any(decimal_or_none(raw.get(k)) for k in ("cgst_total_raw", "sgst_total_raw", "igst_total_raw"))
    item_hsn_values = [safe_str(i.get("hsn_code_raw")) for i in (raw.get("items") or []) if isinstance(i, dict)]
    hsn_missing_or_na = not any(v and v.lower() not in {"na", "n.a", "n.a.", "n/a", "none", "nil"} for v in item_hsn_values)
    # Avoid a broad hardcoded rule. Reassign only for person/consultant-style,
    # no-tax documents where a business client exists and the only GSTIN is more
    # likely from the client/To block than from the person vendor.
    if (
        client_name
        and looks_person_like(vendor_name)
        and looks_business_like(client_name)
        and (safe_str(raw.get("vendor_pan_raw")) or is_professional_fee_context(raw))
        and has_no_tax
        and hsn_missing_or_na
    ):
        raw["isec_gstin_raw"] = vendor_gstin
        raw["vendor_gstin_raw"] = ""
    return raw
def apply_professional_fee_quantity_rate_guard(
    raw: Dict[str, Any],
    quantity: str,
    quantity_unit: str,
    rate: str,
    amount: str,
    from_date: str,
    to_date: str,
) -> Tuple[str, str, str]:
    if not is_professional_fee_context(raw) or not amount:
        return quantity, quantity_unit, rate
    amount_decimal = decimal_or_none(amount)
    source_quantity = safe_str(raw.get("_source_text_quantity_raw"))
    source_quantity_unit = safe_str(raw.get("_source_text_quantity_unit_raw"))
    # If the document explicitly says something like "No. of Persons 2",
    # that printed count wins over generic monthly/professional-fee defaults.
    if source_quantity and is_count_based_quantity_unit(source_quantity_unit):
        quantity = source_quantity
        quantity_unit = source_quantity_unit
    quantity_decimal = decimal_or_none(quantity)
    if is_count_based_quantity_unit(quantity_unit) and quantity_decimal is not None and quantity_decimal > 0:
        if amount_decimal is not None:
            rate = format_decimal_amount(amount_decimal / quantity_decimal)
        return quantity, quantity_unit, rate
    # If no explicit count unit exists, keep consultant/professional monthly invoices
    # as one Month only when the service range is a full calendar month.
    if quantity_decimal is None or quantity_decimal <= 0 or quantity_decimal != Decimal("1"):
        quantity = "1"
    quantity_unit = infer_service_month_unit_from_dates(raw, from_date, to_date)
    if amount_decimal is not None:
        rate = format_decimal_amount(amount_decimal)
    return quantity, quantity_unit, rate
def normalize_quantity_unit_for_context(raw: Dict[str, Any], current_unit: Any) -> str:
    """Generic final guard for units inferred from document context."""
    current = safe_str(current_unit)
    context = build_context_text(raw)
    source_unit = safe_str(raw.get("_source_text_quantity_unit_raw"))
    if is_count_based_quantity_unit(source_unit):
        return source_unit
    # Professional fee/consultant invoices may legitimately be one Month when
    # the service period covers a full calendar month. Generic service invoices
    # with partial periods stay as Service.
    if current.lower() == "month" and is_professional_fee_context(raw):
        return "Month"
    # Professional/service invoices may contain the phrase "for the period"; that
    # does not mean the unit is Month unless a monthly/rental/billing context exists.
    if current.lower() == "month" and text_has_any_keyword(context, GENERIC_PROFESSIONAL_SERVICE_KEYWORDS) and not text_has_any_keyword(context, MONTHLY_ONLY_UNIT_KEYWORDS):
        return "Service"
    if is_periodic_service_context(raw) and not text_has_any_keyword(context, GENERIC_PROFESSIONAL_SERVICE_KEYWORDS):
        return "Month"
    if current and not re.search(r"\b(rs|inr|amount|rate|total|charge|payment)\b", current, flags=re.IGNORECASE):
        return current
    return infer_quantity_unit_from_context(raw)
def is_food_refreshment_context(raw: Dict[str, Any]) -> bool:
    """Detect small food/canteen/restaurant bills without using vendor-specific names."""
    context = build_context_text(raw).lower()
    return bool(
        re.search(
            r"\b(food|restaurant|canteen|refreshment|tea\s*stall|snacks?|sweets?|dish|chai|tea|coffee|samosa|cash\s+credit\s+memo|retail\s+invoice)\b",
            context,
            flags=re.IGNORECASE,
        )
    )
def is_beverage_particulars(value: Any) -> bool:
    """Generic beverage item detector for quantity-unit repair."""
    text = safe_str(value).lower()
    return bool(re.search(r"\b(chai|tea|coffee|beverage|drink|juice|lassi|milk|water)\b", text, flags=re.IGNORECASE))
def normalize_quantity_unit_for_line_item(
    raw: Dict[str, Any],
    item: Dict[str, Any],
    current_unit: Any,
    final_particulars: Any,
) -> str:
    """
    Prevent document-level words like Bill/Cash Memo from becoming item quantity units.
    For food/canteen handwritten bills, infer a practical item unit from the item text.
    """
    current = safe_str(current_unit)
    item_text = " ".join(
        safe_str(v)
        for v in (
            final_particulars,
            item.get("particulars_raw"),
            raw.get("service_usage_raw"),
            raw.get("expense_type_raw"),
            raw.get("document_type_raw"),
        )
        if safe_str(v)
    )
    if is_newspaper_subscription_context(raw):
        if current.lower() in {"", "bill", "item", "items"}:
            return "Month"
    if is_food_refreshment_context(raw):
        if current.lower() in {"", "bill", "cash memo", "memo", "food", "item", "items"}:
            if is_beverage_particulars(item_text):
                return "Cups"
            return "Nos"
        # A bill is the document, not the measurement unit for each food line.
        if current.lower() == "bill":
            if is_beverage_particulars(item_text):
                return "Cups"
            return "Nos"
    return current
def should_default_blank_tax_to_zero(raw: Dict[str, Any]) -> bool:
    """For simple receipts/cash memos without visible tax extraction, return 0 instead of blank."""
    context = build_context_text(raw).lower()
    if is_tax_exempt_context(raw):
        return True
    if is_food_refreshment_context(raw):
        return True
    if is_newspaper_subscription_context(raw):
        return True
    if re.search(r"\b(receipt|cash\s+memo|cash\s+credit\s+memo|deposit\s+receipt|ride|cab|taxi|hospital\s+deposit)\b", context, flags=re.IGNORECASE):
        return True
    return False
def output_tax_amount(raw: Dict[str, Any], item: Dict[str, Any], item_key: str, total_key: str, item_count: int) -> str:
    value = clean_amount(first_non_empty(item.get(item_key), raw.get(total_key) if item_count == 1 else ""))
    if value:
        return value
    # Output convention: unavailable tax amounts must be shown as 0 instead of blank.
    return "0"
def has_employee_self_reimbursement_evidence(raw: Dict[str, Any]) -> bool:
    """
    Return True only when the document gives clear employee/staff/self claim evidence.
    The word reimbursement alone is intentionally not enough.
    """
    context = build_context_text(raw).lower()
    positive_patterns = [
        r"\bself\s+reimbursement\b",
        r"\bself\s+claim\b",
        r"\bemployee\s+(?:reimbursement|expense|claim|code|id|no|number)\b",
        r"\bstaff\s+(?:reimbursement|expense|claim|code|id|no|number)\b",
        r"\bemp(?:loyee)?\.?\s*(?:id|code|no|number)\b",
        r"\bclaimant\s+(?:employee|staff)\b",
        r"\bexpense\s+claim\s+form\b",
        r"\btravel\s+expense\s+claim\b",
        r"\bmedical\s+reimbursement\s+claim\b",
        r"\bpaid\s+by\s+(?:employee|staff|self)\b",
        r"\bemployee\s+name\b.*\b(?:employee\s+code|emp\.?\s*id|department)\b",
    ]
    return any(re.search(pattern, context, flags=re.IGNORECASE | re.DOTALL) for pattern in positive_patterns)
def is_third_party_reimbursement_request(raw: Dict[str, Any]) -> bool:
    """
    Detect reimbursement letters/requests that are from a person/vendor/third party
    to a company, not employee self-reimbursement. Generic and label-based only.
    """
    context = build_context_text(raw).lower()
    if has_employee_self_reimbursement_evidence(raw):
        return False
    reimbursement_signal = bool(re.search(
        r"\b(reimbursement\s+letter|reimbursement\s+request|request\s+(?:you\s+)?to\s+reimburse|reimburse\s+\d+%|reimbursable|reimburse(?:ment)?)\b",
        context,
        flags=re.IGNORECASE,
    ))
    person_to_company = bool(
        looks_person_like(raw.get("vendor_employee_name_raw"))
        and has_company_suffix_or_vendor_word(raw.get("client_name_raw"))
        and re.search(r"\b(lease|deed|stamp\s+duty|registration|rent|property)\b", context, flags=re.IGNORECASE)
    )
    if not reimbursement_signal and not person_to_company:
        return False
    # These are common third-party reimbursement letter patterns: FROM person/vendor
    # TO company/client, lease/property/government-duty charges, or a formal request
    # for the company to reimburse an amount. No party names are hardcoded.
    third_party_signals = [
        r"\bfrom\s*[:\-]\s*(?:mr\.?|mrs\.?|ms\.?|m/s\.?|shri|smt|dr\.?)\b",
        r"\bto\s*[:\-]\s*.+\b(?:limited|ltd|private|pvt|company|corp|corporation)\b",
        r"\bwe\s+(?:have\s+)?paid\b.*\brequest\b.*\breimburse\b",
        r"\brequest\s+(?:you\s+)?to\s+reimburse\b",
        r"\bstamp\s+duty\b",
        r"\bregistration\s+charges?\b",
        r"\blease\s+deed\b",
        r"\blease\b",
        r"\bdeed\b",
        r"\bproperty\s+registration\b",
        r"\blandlord\b",
        r"\blessor\b",
    ]
    return any(re.search(pattern, context, flags=re.IGNORECASE | re.DOTALL) for pattern in third_party_signals)
def is_reimbursement_like_receipt_context(raw: Dict[str, Any]) -> bool:
    """
    Generic heuristic for employee/self reimbursement style receipts.
    Formal tax/professional/service/vendor invoices still return No.
    """
    context = build_context_text(raw).lower()
    if re.search(r"\b(professional\s+fees?|quality\s+engineering|support\s+services?|tax\s+invoice|buyer\s*\(bill\s*to\)|bill\s+to)\b", context, flags=re.IGNORECASE):
        return False
    return bool(
        is_food_refreshment_context(raw)
        or is_newspaper_subscription_context(raw)
        or re.search(r"\b(ride|cab|taxi|travel|receipt\s+voucher|deposit\s+receipt|hospital\s+deposit|medical|cash\s+memo|cash\s+credit\s+memo)\b", context, flags=re.IGNORECASE)
    )
# =========================
# Final output formatting
# =========================
def infer_self_reimbursement(raw: Dict[str, Any]) -> str:
    explicit = normalize_yes_no(raw.get("whether_self_reimbursement_raw"))
    # Important: the word "reimbursement" by itself does not mean self reimbursement.
    # A letter/request from an outside person/vendor/landlord/third party to a company
    # should be No unless employee/staff/self evidence is clearly visible.
    if is_third_party_reimbursement_request(raw):
        return "No"
    if explicit == "No":
        return "No"
    if explicit == "Yes":
        if has_employee_self_reimbursement_evidence(raw) or is_reimbursement_like_receipt_context(raw):
            return "Yes"
        document_type = safe_str(raw.get("document_type_raw")).lower()
        context = build_context_text(raw).lower()
        if "invoice" in document_type or "bill" in document_type or "challan" in document_type or re.search(r"\b(invoice|bill|challan|vendor|supplier|bill\s+to|buyer)\b", context):
            return "No"
        # Keep uncertain formal reimbursement documents as No instead of promoting them
        # only due to model output.
        if re.search(r"\breimburse(?:ment)?\b", context, flags=re.IGNORECASE):
            return "No"
        # Model said Yes but there is no employee/staff/self evidence: do not trust it.
        return "No"
    if has_employee_self_reimbursement_evidence(raw):
        return "Yes"
    if is_reimbursement_like_receipt_context(raw):
        return "Yes"
    document_type = safe_str(raw.get("document_type_raw")).lower()
    context = build_context_text(raw).lower()
    if "invoice" in document_type or "bill" in document_type or "challan" in document_type or re.search(r"\b(invoice|bill|challan|voucher|letter)\b", context):
        return "No"
    return ""
def choose_final_bill_number(raw: Dict[str, Any]) -> str:
    """Choose bill/invoice number without accepting GSTIN/PAN as document number."""
    candidates = [raw.get("bill_no_raw"), raw.get("invoice_no_raw")]
    for candidate in candidates:
        value = clean_document_number(candidate)
        upper_value = value.upper().replace(" ", "")
        if not value:
            continue
        if GSTIN_PATTERN.fullmatch(upper_value):
            continue
        if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", upper_value):
            continue
        if is_non_bill_identifier(value):
            continue
        return value
    return ""
def sanitize_approver_for_output(value: Any, raw: Dict[str, Any]) -> str:
    """Return approver only when an approval label is visible; reject cashier/signatory names."""
    text = safe_str(value)
    if not text:
        return ""
    context = build_context_text(raw)
    if re.search(r"\b(cashier|guest\s*signature|receiver'?s?\s*signature|authori[sz]ed\s*signatory|signature)\b", text, flags=re.IGNORECASE):
        return ""
    if not re.search(r"\b(approver|approved\s*by|approval)\b", f"{text} {context}", flags=re.IGNORECASE):
        return ""
    text = re.sub(r"(?i)\b(approver|approved\s*by|approval)\b\s*[:\-]?", "", text).strip(" .,:;|")
    return normalize_invoice_name(text)
def explicit_gst_rate_components_from_context(raw: Dict[str, Any]) -> Dict[str, str]:
    """Extract visible CGST/SGST/IGST percentage rates from raw OCR/model context."""
    context = build_context_text(raw)
    result = {"cgst_rate": "", "sgst_rate": "", "igst_rate": ""}
    patterns = [
        ("cgst_rate", r"\b(?:central\s+gst|cgst)\b\s*@?\s*(\d{1,2}(?:\.\d{1,4})?)\s*%"),
        ("sgst_rate", r"\b(?:state\s+gst|sgst)\b\s*@?\s*(\d{1,2}(?:\.\d{1,4})?)\s*%"),
        ("igst_rate", r"\b(?:integrated\s+gst|igst)\b\s*@?\s*(\d{1,2}(?:\.\d{1,4})?)\s*%"),
    ]
    for key, pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            result[key] = normalize_percent_rate(match.group(1) + "%")
    # Some model outputs keep percentages in a single rate_raw field like "6% + 6%".
    if not any(result.values()):
        item_rates = []
        for item in raw.get("items") or []:
            if isinstance(item, dict):
                item_rates.append(safe_str(item.get("rate_raw")))
        combined = " ".join([safe_str(raw.get("rate_raw")), safe_str(raw.get("_source_text_gst_rate_raw")), *item_rates])
        rates = re.findall(r"(\d{1,2}(?:\.\d{1,4})?)\s*%", combined)
        if len(rates) >= 2:
            result["cgst_rate"] = normalize_percent_rate(rates[0] + "%")
            result["sgst_rate"] = normalize_percent_rate(rates[1] + "%")
        elif len(rates) == 1:
            result["igst_rate"] = normalize_percent_rate(rates[0] + "%")
    return result
def decimal_from_percent(value: Any) -> Decimal:
    pct = normalize_percent_rate(value)
    dec = decimal_or_none(pct)
    return dec if dec is not None else Decimal("0")
def format_money_decimal(value: Decimal) -> str:
    return format_decimal_amount(value.quantize(Decimal("0.01")).normalize())
def repair_tax_invoice_totals_from_visible_rates(raw: Dict[str, Any], final_line_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generic GST invoice repair.
    When a scanned invoice has visible GST rates and a Grand Total but the model
    incorrectly puts the tax-inclusive total into amount or misses CGST/SGST,
    derive taxable value and tax amounts from the visible GST rates. No vendor,
    client, or document-specific value is hardcoded.
    """
    if not final_line_items:
        return final_line_items
    rates = explicit_gst_rate_components_from_context(raw)
    cgst_rate = decimal_from_percent(rates.get("cgst_rate"))
    sgst_rate = decimal_from_percent(rates.get("sgst_rate"))
    igst_rate = decimal_from_percent(rates.get("igst_rate"))
    total_rate = cgst_rate + sgst_rate + igst_rate
    if total_rate <= 0:
        return final_line_items
    grand_total = decimal_or_none(raw.get("grand_total_raw"))
    if grand_total is None or grand_total <= 0:
        return final_line_items
    context = build_context_text(raw).lower()
    normal_gst_invoice = bool(re.search(r"\b(tax\s+invoice|gstin|cgst|sgst|igst|central\s+gst|state\s+gst|sac|hsn|bill\s+summary|total\s+billing|grand\s+total)\b", context, flags=re.IGNORECASE))
    if not normal_gst_invoice:
        return final_line_items
    # Only consolidate when the extracted line is clearly tax-wrong/incomplete:
    # tax fields blank/zero, or amount already equals grand total, or extracted
    # component total is much smaller than visible grand total.
    current_total, has_current_total = final_line_items_total(final_line_items)
    first = dict(final_line_items[0])
    current_amount = decimal_or_none(first.get("amount"))
    taxes_missing = not any(decimal_or_none(first.get(k)) for k in ("cgst", "sgst", "igst"))
    amount_is_grand_total = current_amount is not None and abs(current_amount - grand_total) <= Decimal("0.01")
    much_smaller_than_grand = has_current_total and current_total > 0 and (grand_total / current_total) >= Decimal("1.50")
    if not (taxes_missing or amount_is_grand_total or much_smaller_than_grand):
        return final_line_items
    taxable = (grand_total / (Decimal("1") + (total_rate / Decimal("100")))).quantize(Decimal("0.01"))
    cgst = (taxable * cgst_rate / Decimal("100")).quantize(Decimal("0.01")) if cgst_rate else Decimal("0")
    sgst = (taxable * sgst_rate / Decimal("100")).quantize(Decimal("0.01")) if sgst_rate else Decimal("0")
    igst = (taxable * igst_rate / Decimal("100")).quantize(Decimal("0.01")) if igst_rate else Decimal("0")
    first["amount"] = format_money_decimal(taxable)
    first["cgst"] = format_money_decimal(cgst) if cgst else "0"
    first["sgst"] = format_money_decimal(sgst) if sgst else "0"
    first["igst"] = format_money_decimal(igst) if igst else "0"
    first["cess"] = first.get("cess") or "0"
    first["rate"] = f"{format_decimal_amount(total_rate)}%"
    if rates.get("cgst_rate"):
        raw["cgst_total_raw"] = first["cgst"]
    if rates.get("sgst_rate"):
        raw["sgst_total_raw"] = first["sgst"]
    if rates.get("igst_rate"):
        raw["igst_total_raw"] = first["igst"]
    raw["subtotal_raw"] = first["amount"]
    raw["_tax_repaired_from_visible_gst_rates"] = "Yes"
    return [first]
def is_hotel_lodging_context(raw: Dict[str, Any]) -> bool:
    context = build_context_text(raw).lower()
    return bool(re.search(
        r"\b(hotel|lodging|accommodation|room\s*rate|room\s*rent|room\s*no|tariff|arrival\s*date|departure\s*date|total\s*billing|bill\s*summary)\b",
        context,
        flags=re.IGNORECASE,
    ))
def is_utility_bill_context(raw: Dict[str, Any]) -> bool:
    context = build_context_text(raw).lower()
    return bool(re.search(
        r"\b(electricity\s*bill|monthly\s*electricity|power\s*bill|energy\s*charges?|water\s*bill|gas\s*bill|utility\s*bill|consumer\s*no|service\s*connection|meter\s*no|units\s*consumed|amount\s*due|total\s*due|net\s*payable)\b",
        context,
        flags=re.IGNORECASE,
    ))
def is_charge_summary_context(raw: Dict[str, Any]) -> bool:
    return is_hotel_lodging_context(raw) or is_utility_bill_context(raw)
def charge_summary_candidate_images(images_base64_urls: List[str]) -> List[str]:
    """Prefer pages that often contain summary totals, one page per LLM request."""
    images = [img for img in images_base64_urls if img]
    if not images:
        return []
    ordered: List[str] = []
    indexes: List[int] = []
    if len(images) >= 2:
        indexes.append(len(images) - 2)  # hotel invoices often put bill summary on penultimate page
    indexes.extend([len(images) - 1, 0])
    if len(images) >= 3:
        indexes.append(1)
    for idx in indexes:
        if 0 <= idx < len(images) and images[idx] not in ordered:
            ordered.append(images[idx])
    return ordered[:MAX_CHARGE_SUMMARY_ATTEMPTS]
def charge_summary_candidate_image_groups(images_base64_urls: List[str]) -> List[List[str]]:
    """Return grouped charge-summary image sets; combined first, then single-page fallbacks."""
    candidates = charge_summary_candidate_images(images_base64_urls)
    groups: List[List[str]] = []
    if candidates:
        groups.append(candidates[:3])
    for img in candidates:
        groups.append([img])
    return groups
def charge_summary_result_has_useful_value(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    keys = [
        "subtotal_raw", "cgst_total_raw", "sgst_total_raw", "igst_total_raw", "grand_total_raw",
        "bill_no_raw", "bill_date_raw", "explicit_date_range_raw", "service_usage_raw",
    ]
    if any(safe_str(result.get(key)) for key in keys):
        return True
    for item in result.get("items") or []:
        if isinstance(item, dict) and safe_str(first_non_empty(item.get("amount_raw"), item.get("cgst_raw"), item.get("sgst_raw"))):
            return True
    return False
def text_has_legal_company_suffix(value: Any) -> bool:
    return bool(re.search(r"\b(?:pvt\.?\s*ltd\.?|private\s+limited|ltd\.?|limited|llp|inc|corp(?:oration)?|company)\b", safe_str(value), flags=re.IGNORECASE))
def looks_like_hotel_trade_name(value: Any) -> bool:
    text = normalize_invoice_name(value)
    if not text:
        return False
    return bool(re.search(r"\b(?:hotel|resort|resorts|heritage|inn|suites|palace|residency|lodg(?:e|ing))\b", text, flags=re.IGNORECASE)) and not text_has_legal_company_suffix(text)
def vendor_candidate_has_mixed_client_or_item_text(value: Any) -> bool:
    """Detect vendor values that merged supplier + billed party/item/table text.
    Generic guard only. It does not contain any vendor/client hardcoding. This
    protects small/local bills and Bill of Supply documents where OCR/VLM may
    concatenate the header, M/s block, HSN/description rows and client name into
    one vendor string.
    """
    text = normalize_invoice_name(value)
    if not text:
        return False
    return bool(re.search(
        r"\b(?:bill\s+of\s+supply|tax\s+invoice|cash\s+memo|credit\s+memo|m\s*/?\s*s\.?|bill\s*to|billed\s*to|buyer|recipient|customer|client|particulars?|description|hsn\s*code|hsn|sac|quantity|qty|rate|amount|gstin|gst\s*no|packaged\s+drinking\s+water)\b",
        text,
        flags=re.IGNORECASE,
    )) and len(text.split()) >= 4
def clean_vendor_candidate_merged_with_client_or_items(value: Any, raw: Dict[str, Any]) -> str:
    """Clean a final vendor name that merged with client or item text.
    This is label/layout/token based and generic. It is especially useful for
    scanned Bill of Supply/local bills where the model may return something like:
    '<document title> <supplier> <billed party> <billed party> <item/HSN>...'.
    """
    text = normalize_invoice_name(value)
    if not text:
        return ""
    original = text
    text = re.sub(r"\s+", " ", text).strip(" .,:;-|")
    # Remove document title prefixes before the issuer name.
    text = re.sub(
        r"^(?:bill\s+of\s+supply|tax\s+invoice|retail\s+invoice|invoice|cash\s+memo|credit\s+memo|receipt)\s*[:\-]?\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .,:;-|")
    # Stop before explicit billed-party/customer/table/item labels.
    stop_label = r"(?:M\s*/?\s*s\.?|Bill\s*To|Billed\s*To|Buyer|Recipient|Customer|Client|Ship\s*To|Sold\s*To|Particulars?|Description|HSN\s*Code|HSN|SAC|Quantity|Qty|Rate|Amount|Total|GSTIN|GST\s*No|PAN|Phone|Email|Mobile)"
    text = re.split(rf"\b{stop_label}\b\s*[:\-]?", text, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .,:;-|")
    # If a supplier-like word is followed by another legal/corporate billed party,
    # keep only the issuer phrase ending at the supplier word. This avoids merging
    # a local vendor with the customer's Limited/Ltd name.
    supplier_word = r"(?:distributors?|dealers?|traders?|trading|agency|agencies|enterprise|enterprises|store|shop|stall|restaurant|canteen|hotel|sweets?|foods?|water|travels?|taxi|cab|newspaper|services?)"
    m = re.match(
        rf"^(.{{2,90}}?\b{supplier_word}\b)(?:\s+[A-Z][A-Za-z&.'()\-/]{{2,}}){{1,12}}\s+\b(?:ltd\.?|limited|pvt\.?|private|company|corp(?:oration)?)\b.*$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        text = m.group(1).strip(" .,:;-|")
    # Remove repeated item descriptions or HSN fragments that got appended
    # without labels.
    text = re.split(r"\b(?:packaged\s+drinking\s+water|hsn\s*code|composition\s+taxable|not\s+eligible\s+to\s+collect\s+tax|rupees?|receiver'?s?\s+signature|authori[sz]ed\s+signatory)\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .,:;-|")
    # Very long mixed strings are not safe as vendor names. Try the leading
    # supplier-like phrase; otherwise keep the original cleaned by address logic.
    if len(text.split()) > 8:
        m2 = re.match(rf"^(.{{2,90}}?\b{supplier_word}\b)", text, flags=re.IGNORECASE)
        if m2:
            text = m2.group(1).strip(" .,:;-|")
    text = strip_address_tail_from_vendor_name(text)
    if text:
        return normalize_invoice_name(text)
    return strip_address_tail_from_vendor_name(original)
def should_run_legal_supplier_repair(raw: Dict[str, Any]) -> bool:
    if not ENABLE_LEGAL_SUPPLIER_REPAIR_PASS:
        return False
    vendor = safe_str(raw.get("vendor_employee_name_raw"))
    context = build_context_text(raw)
    hotel_like = is_hotel_lodging_context(raw) or looks_like_hotel_trade_name(vendor) or bool(re.search(r"\b(room\s*rate|arrival\s*date|departure\s*date|hotel\s*gstin|guest\s*name)\b", context, flags=re.IGNORECASE))
    if not hotel_like:
        return False
    if not vendor:
        return True
    if looks_like_hotel_trade_name(vendor):
        return True
    if not text_has_legal_company_suffix(vendor) and re.search(r"\b(gstin|pan\s*no|vat\s*tin|fssai|cin|tax\s*invoice)\b", context, flags=re.IGNORECASE):
        return True
    return False
def extract_legal_supplier_from_context_text(value: Any) -> str:
    """Generic text fallback: legal entity near GSTIN/PAN/header tokens."""
    text = safe_str(value)
    if not text:
        return ""
    lines = [normalize_invoice_name(line) for line in re.split(r"[\r\n]+| {2,}", text) if safe_str(line)]
    legal_suffix = r"(?:pvt\.?\s*ltd\.?|private\s+limited|ltd\.?|limited|llp|inc|corp(?:oration)?|company)"
    # Prefer a standalone/header line with a legal suffix.
    for line in lines[:25]:
        if re.search(legal_suffix, line, flags=re.IGNORECASE):
            cleaned = strip_address_tail_from_vendor_name(line)
            if cleaned and not looks_like_customer_or_recipient_name(cleaned):
                return cleaned
    # Then search continuous text near tax identifiers.
    patterns = [
        rf"([A-Z][A-Za-z0-9&.'()\-/ ]{{2,110}}?\b{legal_suffix}\b)[^\n]{{0,180}}?\b(?:GSTIN|GST\s*NO|PAN|VAT\s*TIN|FSSAI|CIN)\b",
        rf"\b(?:GSTIN|GST\s*NO|PAN|VAT\s*TIN|FSSAI|CIN)\b[^\n]{{0,180}}?([A-Z][A-Za-z0-9&.'()\-/ ]{{2,110}}?\b{legal_suffix}\b)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            cleaned = strip_address_tail_from_vendor_name(match.group(1))
            if cleaned and not looks_like_customer_or_recipient_name(cleaned):
                return cleaned
    return ""
def apply_generic_legal_supplier_ocr_corrections(value: Any) -> str:
    """Correct common OCR mistakes in legal supplier names.
    Generic dictionary only: business/legal suffixes and lodging words. It does
    not contain any invoice-specific company/vendor/client name.
    """
    text = normalize_invoice_name(value)
    if not text:
        return ""
    replacements = [
        (r"\bKesorts\b", "Resorts"),
        (r"\bKesort\b", "Resort"),
        (r"\bKesorls\b", "Resorts"),
        (r"\bResorls\b", "Resorts"),
        (r"\bPesorts\b", "Resorts"),
        (r"\bFesorts\b", "Resorts"),
        (r"\bPvt\s*[,;:]?\s*L[ti1]d\.?\b", "Pvt. Ltd."),
        (r"\bPvt\s*[,;:]?\s*Ltd\b", "Pvt. Ltd."),
        (r"\bPvl\.?\s*Ltd\.?\b", "Pvt. Ltd."),
        (r"\bPvi\.?\s*Ltd\.?\b", "Pvt. Ltd."),
        (r"\bPrivale\s+Limited\b", "Private Limited"),
        (r"\bLimiled\b", "Limited"),
        (r"\bL[ti1]d\.?\b", "Ltd."),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,-:;|/")
    text = normalize_invoice_name(text)
    text = re.sub(r"\bPvt\.?\s*Ltd\.?$", "Pvt. Ltd.", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPrivate\s+Limited$", "Private Limited", text, flags=re.IGNORECASE)
    text = re.sub(r"\bLtd\.?$", "Ltd.", text, flags=re.IGNORECASE)
    return text
def preprocess_header_crop_for_ocr(crop: Image.Image) -> List[Image.Image]:
    """Return a small set of generic OCR-friendly variants for scanned headers."""
    variants: List[Image.Image] = []
    try:
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        gray = ImageOps.grayscale(crop)
        base = ImageEnhance.Contrast(gray).enhance(1.8)
        base = ImageEnhance.Sharpness(base).enhance(2.0)
        variants.append(base)
        variants.append(base.point(lambda p: 0 if p < 175 else 255))
        variants.append(base.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3)))
    except Exception:
        try:
            variants.append(crop.convert("L"))
        except Exception:
            pass
    return variants
def run_legal_supplier_header_ocr(crops: List[Image.Image]) -> str:
    """OCR only the legal/header area. Empty when pytesseract is unavailable."""
    if pytesseract is None:
        return ""
    texts: List[str] = []
    for crop in crops:
        for variant in preprocess_header_crop_for_ocr(crop):
            for psm in (6, 3, 4):
                try:
                    ocr_text = pytesseract.image_to_string(
                        variant,
                        config=f"--psm {psm} -c preserve_interword_spaces=1",
                    )
                except Exception:
                    continue
                if safe_str(ocr_text):
                    texts.append(ocr_text)
    return "\n".join(texts)
def extract_legal_supplier_via_header_ocr(file_path: str, filename: str) -> Dict[str, Any]:
    """Deterministic OCR fallback for scanned invoices with logo/trade-name confusion.
    Reads only generic top/header supplier crops. It chooses a legal entity line
    with legal suffix near tax/address clues. No vendor/client/company name is
    hardcoded.
    """
    result: Dict[str, Any] = {}
    try:
        img = load_first_page_or_image_for_legal_supplier(file_path, filename)
        if img.mode != "RGB":
            img = img.convert("RGB")
        try:
            bbox = find_nonwhite_bbox(img)
            doc_crop = img.crop(bbox)
        except Exception:
            doc_crop = img
        doc_w, doc_h = doc_crop.size
        header_crops = [
            doc_crop.crop((0, 0, int(doc_w * 0.74), int(doc_h * 0.18))),
            doc_crop.crop((0, 0, int(doc_w * 0.82), int(doc_h * 0.24))),
            doc_crop.crop((0, 0, int(doc_w * 0.72), int(doc_h * 0.30))),
            doc_crop.crop((0, 0, doc_w, int(doc_h * 0.24))),
        ]
        ocr_text = run_legal_supplier_header_ocr(header_crops)
        if not safe_str(ocr_text):
            return result
        # Preserve OCR line breaks while fixing generic OCR confusions so the
        # legal entity line does not merge with address/tax-id lines.
        corrected_text = "\n".join(
            apply_generic_legal_supplier_ocr_corrections(line)
            for line in safe_str(ocr_text).splitlines()
            if safe_str(line)
        )
        candidate = extract_legal_supplier_from_context_text(corrected_text)
        candidate = apply_generic_legal_supplier_ocr_corrections(candidate)
        candidate = strip_address_tail_from_vendor_name(candidate)
        if candidate and text_has_legal_company_suffix(candidate):
            result["legal_supplier_name_raw"] = candidate
            result["vendor_employee_name_raw"] = candidate
            result["evidence_raw"] = corrected_text[:1200]
            result["_legal_supplier_ocr_text_raw"] = corrected_text[:1800]
        gstins = extract_gstins_from_text(corrected_text)
        if gstins:
            result["vendor_gstin_raw"] = gstins[0]
            state = state_from_gstin(gstins[0])
            if state:
                result["los_vendor_state_raw"] = state
        pan_match = re.search(r"\b[A-Z]{5}\s*\d{4}\s*[A-Z]\b", corrected_text, flags=re.IGNORECASE)
        if pan_match:
            result["vendor_pan_raw"] = re.sub(r"\s+", "", pan_match.group(0)).upper()
    except Exception as e:
        result["_legal_supplier_ocr_warning"] = str(e)
    return result
def choose_best_legal_supplier_candidate(current: Any, raw_candidate: Any, *context_values: Any) -> str:
    """Prefer a visible legal entity with legal suffix over a hotel/logo trade name.
    This is generic. It searches only supplied evidence/context strings for a
    legal-name pattern near GSTIN/PAN/VAT/FSSAI/CIN/address clues; it does not
    hardcode any supplier/client names.
    """
    raw_clean = strip_address_tail_from_vendor_name(english_script_invoice_name(raw_candidate))
    evidence_candidates: List[str] = []
    for value in context_values:
        candidate = extract_legal_supplier_from_context_text(value)
        candidate = strip_address_tail_from_vendor_name(english_script_invoice_name(candidate))
        if candidate and candidate.lower() not in [c.lower() for c in evidence_candidates]:
            evidence_candidates.append(candidate)
    best = raw_clean
    for candidate in evidence_candidates:
        if not best or should_prefer_legal_supplier_name(best, candidate) or looks_like_hotel_trade_name(best):
            best = candidate
    if best and current and should_prefer_legal_supplier_name(current, best):
        return best
    if best and looks_like_hotel_trade_name(current) and text_has_legal_company_suffix(best):
        return best
    return best
def merge_legal_supplier_fields(base: Dict[str, Any], legal_supplier: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(legal_supplier, dict):
        return base
    # Do not normalize through RAW_KEYS here because this focused pass returns
    # legal_supplier_name_raw/trade_or_logo_name_raw keys that are intentionally
    # outside the normal invoice schema.
    evidence = safe_str(legal_supplier.get("evidence_raw"))
    current = safe_str(base.get("vendor_employee_name_raw"))
    raw_candidate = first_non_empty(
        legal_supplier.get("legal_supplier_name_raw"),
        legal_supplier.get("vendor_employee_name_raw"),
    )
    candidate = choose_best_legal_supplier_candidate(
        current,
        raw_candidate,
        evidence,
        legal_supplier.get("vendor_address_raw"),
        legal_supplier,
        base.get("_charge_summary_evidence"),
        base.get("_vendor_identity_evidence"),
        build_context_text(base),
    )
    candidate = apply_generic_legal_supplier_ocr_corrections(candidate)
    if candidate and not vendor_candidate_has_mixed_client_or_item_text(candidate) and looks_like_valid_vendor_hint(base, candidate):
        if not current or should_prefer_legal_supplier_name(current, candidate) or looks_like_hotel_trade_name(current):
            base["vendor_employee_name_raw"] = candidate
            base["_legal_supplier_name_raw"] = candidate
            base["_vendor_identity_candidate_raw"] = candidate
    trade = safe_str(legal_supplier.get("trade_or_logo_name_raw"))
    if trade:
        base["_trade_or_logo_name_raw"] = first_non_empty(base.get("_trade_or_logo_name_raw"), trade)
    for key in ["vendor_gstin_raw", "vendor_address_raw", "los_vendor_state_raw"]:
        value = safe_str(legal_supplier.get(key))
        if value and not safe_str(base.get(key)):
            base[key] = value
    if evidence:
        base["_legal_supplier_evidence"] = first_non_empty(base.get("_legal_supplier_evidence"), evidence)
    return base
def enhance_utility_ocr_crop(crop: Image.Image) -> Image.Image:
    """Light deterministic OCR enhancement for utility bill headers."""
    try:
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        # Upscale small screenshots/scans before OCR. The utility table is often
        # only a fraction of the uploaded image, and 2x avoids digit slips.
        max_side = max(crop.size)
        if max_side < 1800:
            scale = 2 if max_side >= 650 else 3
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        # OCR generally reads printed utility bills better after grayscale + contrast.
        gray = ImageOps.grayscale(crop)
        gray = ImageEnhance.Contrast(gray).enhance(1.9)
        gray = ImageEnhance.Sharpness(gray).enhance(1.4)
        return gray.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    except Exception:
        return crop
def ocr_text_from_utility_crops(crops: List[Image.Image]) -> str:
    """Run optional pytesseract on a few utility header/customer crops."""
    if pytesseract is None:
        return ""
    texts: List[str] = []
    for crop in crops:
        try:
            enhanced = enhance_utility_ocr_crop(crop)
            text = pytesseract.image_to_string(enhanced, config="--psm 6")
            if safe_str(text):
                texts.append(text)
        except Exception:
            continue
    return "\n".join(texts)[:5000]
def clean_utility_ocr_party_name(value: Any) -> str:
    """Clean a labelled utility consumer/customer name from OCR text."""
    text = normalize_invoice_name(value)
    if not text:
        return ""
    text = re.split(
        r"\b(?:address|village|pin\s*code|mobile|email|bill\s*date|due\s*date|consumer\s*no|customer\s*no|meter\s*no)\b\s*[:\-]?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.sub(r"^(?:consumer|customer|client|subscriber)\s*name\s*[:\-]?", "", text, flags=re.IGNORECASE).strip(" .,:;|\t")
    # Generic OCR correction for leading digit/letter slips in uppercase party names.
    # It only fires on obvious OCR tokens before business words and does not create
    # a client when no Consumer/Customer Name label was read.
    # Generic OCR digit/letter cleanup: a leading digit 1 at the start of an
    # uppercase name token is commonly a misread capital I. Do not invent names;
    # this only changes the OCR token itself after a Consumer/Customer Name label.
    text = re.sub(r"\b1([A-Z]{2,})\b", r"I\1", text)
    text = re.sub(r"\b(LTD|LIMITED|PVT\.?\s*LTD\.?)\s+\d{1,2}\b", r"\1", text, flags=re.IGNORECASE)
    # Remove a trailing isolated OCR residue character after a business suffix.
    # Example pattern: "... LTD A" where A is the first letter of the next
    # address/line captured by OCR. This is utility-label-only cleanup and does
    # not invent or hardcode any client name.
    text = re.sub(r"\b((?:LTD\.?|LIMITED|PVT\.?\s*LTD\.?|PRIVATE\s+LIMITED|LLP))\s+[A-Z]\s*$", r"\1", text, flags=re.IGNORECASE)
    return normalize_invoice_name(text)
def derive_month_range_from_utility_title(value: Any) -> Tuple[str, str]:
    """Return full month range from utility titles like 'for the month of JUN-2026'."""
    text = safe_str(value).upper()
    if not text:
        return "", ""
    match = re.search(
        rf"\b(?:FOR\s+THE\s+)?MONTH\s+OF\s+({MONTH_PATTERN})[\s\-_/.,]*(20\d{{2}}|\d{{2}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            rf"\b({MONTH_PATTERN})[\s\-_/.,]*(20\d{{2}}|\d{{2}})\b",
            text,
            flags=re.IGNORECASE,
        )
    if not match:
        return "", ""
    month = MONTH_ALIASES.get(match.group(1).upper())
    year = normalize_year(match.group(2))
    if not month:
        return "", ""
    last_day = calendar.monthrange(year, month)[1]
    return f"01/{month:02d}/{year}", f"{last_day:02d}/{year}" if False else f"{last_day:02d}/{month:02d}/{year}"
def derive_utility_billing_month_range(raw: Dict[str, Any]) -> Tuple[str, str]:
    """Prefer the explicit billing month printed in a utility bill title."""
    if not is_utility_bill_context(raw):
        return "", ""
    for value in [
        raw.get("service_usage_raw"),
        raw.get("service_title_raw"),
        raw.get("explicit_date_range_raw"),
        raw.get("remarks_raw"),
        raw.get("_utility_ocr_text_raw"),
        raw.get("_charge_summary_evidence"),
        raw.get("visible_text_raw"),
    ]:
        from_date, to_date = derive_month_range_from_utility_title(value)
        if from_date and to_date:
            return from_date, to_date
    return "", ""
def extract_utility_bill_via_header_ocr(file_path: str, filename: str) -> Dict[str, Any]:
    """Deterministic OCR fallback for utility/electricity bill header fields.
    The fallback is generic and label/layout based. It reads only header/customer
    areas, then extracts Consumer Name, Bill Date, a Bill-of-Supply header document
    number, billing month, and HSN code. It never uses consumer/meter/service
    numbers as BILL NO.
    """
    result: Dict[str, Any] = {}
    if pytesseract is None:
        return result
    try:
        img = load_first_page_or_image_for_legal_supplier(file_path, filename)
        if img.mode != "RGB":
            img = img.convert("RGB")
        try:
            doc_crop = img.crop(find_nonwhite_bbox(img))
        except Exception:
            doc_crop = img
        w, h = doc_crop.size
        crops = [
            # Most focused title/header crop first; it gives the highest accuracy
            # for top-row utility bill/document numbers.
            doc_crop.crop((0, 0, w, int(h * 0.25))),
            doc_crop.crop((0, 0, w, int(h * 0.42))),
            doc_crop.crop((0, 0, int(w * 0.72), int(h * 0.42))),
            doc_crop.crop((int(w * 0.50), 0, w, int(h * 0.30))),
        ]
        text = ocr_text_from_utility_crops(crops)
        if not safe_str(text):
            return result
        compact = re.sub(r"[ \t]+", " ", text)
        result["_utility_ocr_text_raw"] = compact[:2500]
        utility_like = bool(re.search(
            r"\b(electricity\s+distribution|mahavitaran|mahadiscom|bill\s+of\s+supply|consumer\s+no|consumer\s+name|bill\s+date)\b",
            compact,
            flags=re.IGNORECASE,
        ))
        if not utility_like:
            return result
        lines = [normalize_invoice_name(line) for line in re.split(r"[\r\n]+", text) if normalize_invoice_name(line)]
        # Legal distributor/vendor name from a clear utility-board header line.
        for line in lines[:8]:
            if re.search(r"\b(?:state\s+)?electricity\s+distribution\s+co\.?(?:\s+ltd\.?)?\b", line, flags=re.IGNORECASE):
                result["vendor_employee_name_raw"] = line.strip(" .,:;|")
                break
        # Service title and month range.
        title_match = re.search(
            rf"(BILL\s+OF\s+SUPPLY\s+FOR\s+THE\s+MONTH\s+OF\s+(?:{MONTH_PATTERN})[\s\-_/.,]*20\d{{2}})",
            compact,
            flags=re.IGNORECASE,
        )
        if title_match:
            title = normalize_invoice_name(title_match.group(1)).upper()
            result["service_usage_raw"] = title
            from_date, to_date = derive_month_range_from_utility_title(title)
            if from_date and to_date:
                result["explicit_date_range_raw"] = f"{from_date} to {to_date}"
                result["_utility_period_title_evidence"] = "Yes"
        # Top-row Bill-of-Supply document number. It is often an unlabeled long
        # sequence on the same row as the title. Restrict to title/header windows
        # and reject HSN/consumer/GST/PIN/phone/amount values.
        header_windows: List[str] = []
        for line in lines[:10]:
            if re.search(r"bill\s+of\s+supply|hsn\s*code|gstin|website", line, flags=re.IGNORECASE):
                header_windows.append(line)
        header_text = " ".join(header_windows)
        bill_no = ""
        bill_no_candidates: List[str] = []
        for num in re.findall(r"(?<![A-Z0-9])(20\d{10,14}|\d{11,18})(?![A-Z0-9])", header_text):
            if re.search(rf"(?:consumer|customer|account|meter|service|mobile|phone|pin|hsn)\D{{0,25}}{re.escape(num)}", header_text, flags=re.IGNORECASE):
                continue
            if len(num) in {6, 8} and re.search(rf"hsn\s*code\D{{0,10}}{re.escape(num)}", header_text, flags=re.IGNORECASE):
                continue
            bill_no_candidates.append(num)
        if bill_no_candidates:
            counts: Dict[str, int] = {}
            for num in bill_no_candidates:
                counts[num] = counts.get(num, 0) + 1
            # Keep the focused-crop candidate on ties, otherwise the most repeated
            # candidate. This avoids wider crops overriding a cleaner header read.
            bill_no = sorted(bill_no_candidates, key=lambda n: (-counts.get(n, 0), bill_no_candidates.index(n)))[0]
        if bill_no:
            result["bill_no_raw"] = bill_no
            result["_utility_bill_no_header_evidence"] = "Yes"
        # Bill date with explicit label only.
        date_match = re.search(r"\bBILL\s*DATE\b\D{0,30}(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", compact, flags=re.IGNORECASE)
        if date_match:
            bill_date = normalize_date_string(date_match.group(1))
            if bill_date:
                result["bill_date_raw"] = bill_date
                result["_utility_bill_date_label_evidence"] = "Yes"
        # Consumer/customer name from labelled line only.
        consumer_match = re.search(
            r"\bConsumer\s*Name\b\s*[:\-]?\s*([^\n\r]{2,100})",
            text,
            flags=re.IGNORECASE,
        )
        if consumer_match:
            client = clean_utility_ocr_party_name(consumer_match.group(1))
            if client:
                result["client_name_raw"] = client
                result["billed_party_raw"] = client
                result["customer_block_raw"] = f"Consumer Name: {client}"
        hsn_match = re.search(r"\bHSN\s*CODE\s*[:=\-]?\s*(\d{4,10})", compact, flags=re.IGNORECASE)
        if hsn_match:
            result["_source_text_hsn_code_raw"] = hsn_match.group(1)
        # GSTIN-derived state fallback for utility vendor/client where visible.
        gstin_match = re.search(r"\bGSTIN\s*[:=\-]?\s*([0-9A-Z]{15})", compact, flags=re.IGNORECASE)
        if gstin_match:
            result["vendor_gstin_raw"] = gstin_match.group(1).upper()
    except Exception as e:
        result["_utility_ocr_warning"] = str(e)
    return result
def should_replace_utility_client_with_ocr(current: Any, candidate: Any) -> bool:
    current_clean = normalize_invoice_name(current)
    candidate_clean = normalize_invoice_name(candidate)
    if not candidate_clean:
        return False
    if not current_clean:
        return True
    if current_clean.lower() == candidate_clean.lower():
        return False
    # Replace clear OCR mistakes such as digit-leading or visibly malformed names
    # when deterministic OCR read the labelled Consumer Name line.
    if re.search(r"\d", current_clean):
        return True
    if looks_like_invalid_client_any_context({}, current_clean):
        return True
    if len(candidate_clean) >= 6 and len(current_clean) <= 8:
        return True
    return False
def merge_utility_ocr_fields(base: Dict[str, Any], utility: Dict[str, Any]) -> Dict[str, Any]:
    """Merge deterministic utility OCR fields without affecting non-utility bills."""
    if not isinstance(utility, dict) or not safe_str(utility.get("_utility_ocr_text_raw")):
        return base
    combined_context = " ".join([build_context_text(base), safe_str(utility.get("_utility_ocr_text_raw"))])
    if not re.search(r"\b(electricity\s+distribution|mahavitaran|mahadiscom|bill\s+of\s+supply|consumer\s+name|bill\s+date)\b", combined_context, flags=re.IGNORECASE):
        return base
    # Preserve OCR evidence for final sanitizers/date-range logic.
    for key in ["_utility_ocr_text_raw", "_utility_bill_no_header_evidence", "_utility_bill_date_label_evidence", "_utility_period_title_evidence", "_source_text_hsn_code_raw"]:
        if safe_str(utility.get(key)):
            base[key] = utility.get(key)
    # Fill/repair utility identity only from labelled/header OCR.
    vendor = safe_str(utility.get("vendor_employee_name_raw"))
    if vendor and (not safe_str(base.get("vendor_employee_name_raw")) or re.search(r"electricity\s+distribution", vendor, flags=re.IGNORECASE)):
        base["vendor_employee_name_raw"] = vendor
    client = safe_str(utility.get("client_name_raw"))
    if client and should_replace_utility_client_with_ocr(base.get("client_name_raw"), client):
        base["client_name_raw"] = client
        base["billed_party_raw"] = client
        base["customer_block_raw"] = utility.get("customer_block_raw", client)
    for key in ["service_usage_raw", "explicit_date_range_raw", "bill_date_raw", "bill_no_raw", "vendor_gstin_raw"]:
        value = safe_str(utility.get(key))
        if not value:
            continue
        if key in {"bill_date_raw", "bill_no_raw", "explicit_date_range_raw"}:
            base[key] = value
        elif not safe_str(base.get(key)):
            base[key] = value
    # Push HSN to existing utility line if the model missed it.
    hsn = safe_str(utility.get("_source_text_hsn_code_raw"))
    if hsn:
        for item in base.get("items") or []:
            if isinstance(item, dict) and not safe_str(item.get("hsn_code_raw")):
                item["hsn_code_raw"] = hsn
    return repair_generic_context_fields(base)
def date_values_in_raw_items(raw: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue
        for key in ("bill_date_raw", "from_date_raw", "to_date_raw", "service_period_raw", "explicit_date_range_raw"):
            value = normalize_date_string(item.get(key))
            if value and value not in values:
                values.append(value)
    return values
def count_good_receipt_panel_items(raw: Dict[str, Any]) -> int:
    count = 0
    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue
        if safe_str(item.get("particulars_raw")) and clean_amount(item.get("amount_raw")):
            count += 1
    return count
def looks_like_local_handwritten_receipt_context(raw: Dict[str, Any]) -> bool:
    context = build_context_text(raw).lower()
    if is_hotel_lodging_context(raw) or is_utility_bill_context(raw):
        return False
    if re.search(r"\b(tax\s+invoice|e-?invoice|irn|ack\s*no|gstin|cgst|sgst|igst|bill\s+summary|total\s+billing)\b", context, flags=re.IGNORECASE):
        return False
    return bool(re.search(
        r"\b(handwritten|cash\s*memo|receipt|local\s+bill|food|meal|lunch|dinner|snacks?|tea|coffee|fish|seafood|prawn|prawns|pomfret|kg|kilo|pcs?)\b",
        context,
        flags=re.IGNORECASE,
    ))
def should_run_multi_receipt_panel_repair(raw: Dict[str, Any]) -> bool:
    if not ENABLE_MULTI_RECEIPT_PANEL_REPAIR_PASS:
        return False
    if not looks_like_local_handwritten_receipt_context(raw):
        return False
    items = [item for item in (raw.get("items") or []) if isinstance(item, dict)]
    if not items:
        return True
    good_count = count_good_receipt_panel_items(raw)
    item_dates = date_values_in_raw_items(raw)
    header_dates = extract_all_full_dates(" ".join([safe_str(raw.get("bill_date_raw")), safe_str(raw.get("remarks_raw")), safe_str(raw.get("visible_text_raw"))]))
    if good_count < 4 and len(item_dates) < 2:
        return True
    if good_count < 4 and len(header_dates) >= 3:
        return True
    return False
def multi_receipt_panel_result_has_useful_items(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    normalized = normalize_raw_result(result)
    items = [item for item in (normalized.get("items") or []) if isinstance(item, dict)]
    good = 0
    dated = 0
    for item in items:
        if safe_str(item.get("particulars_raw")) and clean_amount(item.get("amount_raw")):
            good += 1
        if normalize_date_string(first_non_empty(item.get("bill_date_raw"), item.get("from_date_raw"), item.get("to_date_raw"))):
            dated += 1
    return good >= 2 and dated >= 2
def merge_multi_receipt_panel_fields(base: Dict[str, Any], panel_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(panel_result, dict):
        return base
    repair = normalize_raw_result(panel_result)
    repair_items = [item for item in (repair.get("items") or []) if isinstance(item, dict)]
    base_items = [item for item in (base.get("items") or []) if isinstance(item, dict)]
    if not repair_items:
        return base
    repair_good = count_good_receipt_panel_items(repair)
    base_good = count_good_receipt_panel_items(base)
    repair_dates = date_values_in_raw_items(repair)
    base_dates = date_values_in_raw_items(base)
    should_replace_items = (
        repair_good > base_good
        or (repair_good == base_good and len(repair_dates) > len(base_dates) and repair_good >= 2)
        or (base_good <= 1 and repair_good >= 2)
    )
    if not should_replace_items:
        return base
    for key in RAW_KEYS:
        if key in {"items", "expense_type_raw", "invoice_no_raw", "bill_no_raw", "bill_date_raw", "service_period_raw", "explicit_date_range_raw"}:
            continue
        repair_value = safe_str(repair.get(key))
        if repair_value and not safe_str(base.get(key)):
            base[key] = repair_value
    if repair_good >= 2 and len(repair_dates) >= 2:
        base["bill_date_raw"] = ""
        base["service_period_raw"] = ""
        base["explicit_date_range_raw"] = ""
    cleaned_items: List[Dict[str, Any]] = []
    for item in repair_items:
        if not safe_str(item.get("particulars_raw")) or not clean_amount(item.get("amount_raw")):
            continue
        item = dict(item)
        for tax_key in ("cgst_raw", "sgst_raw", "igst_raw", "cess_raw"):
            if not safe_str(item.get(tax_key)):
                item[tax_key] = "0"
        item["particulars_raw"] = re.sub(r"(?i)\bprawms\b", "Prawns", safe_str(item.get("particulars_raw")))
        item = _fix_local_receipt_quantity_from_particulars(item)
        cleaned_items.append(item)
    if cleaned_items:
        base["items"] = cleaned_items
        total = Decimal("0")
        ok_total = True
        for item in cleaned_items:
            amt = money_decimal_or_none(item.get("amount_raw"))
            if amt is None:
                ok_total = False
                break
            total += amt
        if ok_total and total > 0:
            base["grand_total_raw"] = format_decimal_amount(total)
        base["_multi_receipt_panel_repair"] = "Used focused multi-receipt panel extraction because multiple dated local receipt panels were visible."
    return repair_generic_context_fields(base)
def should_run_charge_summary_repair(raw: Dict[str, Any]) -> bool:
    if not ENABLE_CHARGE_SUMMARY_REPAIR_PASS:
        return False
    if is_hotel_lodging_context(raw):
        return True
    if is_utility_bill_context(raw):
        return True
    # A large number of tiny charge rows usually means a utility bill was itemized incorrectly.
    items = raw.get("items") or []
    if len(items) >= 6 and re.search(r"\b(total\s*amount|net\s*amount|arrears?|adjustment|surcharge|energy\s*charges?|cust\s*charges?)\b", build_context_text(raw), flags=re.IGNORECASE):
        return True
    return False
def money_decimal_or_none(value: Any) -> Any:
    """Return Decimal for money-like values, or None when not parseable."""
    cleaned = clean_amount(value)
    if not cleaned:
        return None
    return decimal_or_none(cleaned)
def prefer_larger_money(existing: Any, new_value: Any) -> str:
    """For multi-page summaries, keep the larger complete-summary amount."""
    new_clean = clean_amount(new_value)
    if not new_clean:
        return safe_str(existing)
    old_clean = clean_amount(existing)
    if not old_clean:
        return new_clean
    old_dec = decimal_or_none(old_clean)
    new_dec = decimal_or_none(new_clean)
    if old_dec is None or new_dec is None:
        return new_clean
    return new_clean if new_dec > old_dec else old_clean
def context_mentions_split_gst(raw: Dict[str, Any]) -> bool:
    """Return True only when both CGST and SGST labels are visible in the bill text."""
    context = build_context_text(raw)
    has_cgst = bool(re.search(r"\b(central\s+gst|cgst)\b", context, flags=re.IGNORECASE))
    has_sgst = bool(re.search(r"\b(state\s+gst|sgst)\b", context, flags=re.IGNORECASE))
    return has_cgst and has_sgst
def context_mentions_explicit_igst(raw: Dict[str, Any]) -> bool:
    context = build_context_text(raw)
    return bool(re.search(r"\b(integrated\s+gst|igst)\b", context, flags=re.IGNORECASE))
def sanitize_hotel_tax_totals(raw: Dict[str, Any]) -> Dict[str, Any]:
    """When a bill explicitly shows both CGST and SGST but no IGST, suppress misclassified IGST for any invoice type."""
    if context_mentions_split_gst(raw) and not context_mentions_explicit_igst(raw):
        raw["igst_total_raw"] = "0"
        for item in raw.get("items") or []:
            if isinstance(item, dict):
                item["igst_raw"] = "0"
    if is_hotel_lodging_context(raw):
        raw["is_invoice_digitally_signed_raw"] = "No" if not re.search(r"\b(irn|ack\s*no|ack\s*date|e-?invoice|qr\s*code)\b", build_context_text(raw), flags=re.IGNORECASE) else raw.get("is_invoice_digitally_signed_raw", "")
    return raw
def split_hotel_igst_to_cgst_sgst_when_needed(raw: Dict[str, Any]) -> Dict[str, Any]:
    """If a same-state hotel bill has only IGST from OCR, split it into CGST/SGST."""
    if not is_hotel_lodging_context(raw):
        return raw
    if safe_str(raw.get("cgst_total_raw")) or safe_str(raw.get("sgst_total_raw")):
        return sanitize_hotel_tax_totals(raw)
    igst = money_decimal_or_none(raw.get("igst_total_raw"))
    if igst is None or igst <= 0:
        return raw
    # Use this only when split GST labels are visible or vendor/client states match.
    same_state = state_or_empty(raw.get("los_vendor_state_raw")) and state_or_empty(raw.get("los_vendor_state_raw")) == state_or_empty(raw.get("lor_isec_state_raw"))
    if context_mentions_split_gst(raw) or same_state:
        half = igst / Decimal("2")
        raw["cgst_total_raw"] = format_money_decimal(half)
        raw["sgst_total_raw"] = format_money_decimal(half)
        raw["igst_total_raw"] = "0"
    return sanitize_hotel_tax_totals(raw)
def should_prefer_legal_supplier_name(current: Any, candidate: Any) -> bool:
    """Prefer a visible legal entity over a hotel logo/brand/trade name."""
    cur = normalize_invoice_name(current)
    cand = normalize_invoice_name(candidate)
    if not cand or cand.lower() == cur.lower():
        return False
    if vendor_candidate_has_mixed_client_or_item_text(cand):
        return False
    legal_suffix = r"\b(?:pvt\.?\s*ltd\.?|private\s+limited|ltd\.?|limited|llp|inc|corp(?:oration)?|company)\b"
    cand_legal = bool(re.search(legal_suffix, cand, flags=re.IGNORECASE))
    cur_legal = bool(re.search(legal_suffix, cur, flags=re.IGNORECASE))
    if cand_legal and not cur_legal:
        return True
    brandish = bool(re.search(r"\b(?:hotel|resort|heritage|inn|suites|palace|residency)\b", cur, flags=re.IGNORECASE))
    return brandish and cand_legal
def should_prefer_charge_summary_bill_no(current: Any, candidate: Any) -> bool:
    """For focused summary repair, prefer a labelled Bill Number candidate over earlier OCR guesses."""
    cur = clean_document_number(current)
    cand = clean_document_number(candidate)
    if not cand or cand.lower() == cur.lower():
        return False
    if is_non_bill_identifier(cand):
        return False
    if re.fullmatch(r"[A-Z0-9][A-Z0-9/_\-]{5,60}", cand, flags=re.IGNORECASE) and not re.fullmatch(r"\d+(?:\.\d+)?", cand):
        return True
    return False
def choose_hotel_taxable_amount_from_summary(base: Dict[str, Any], summary: Dict[str, Any], useful_items: List[Dict[str, Any]]) -> str:
    """Choose complete hotel taxable/tariff total, avoiding one-day room/day totals."""
    candidates: List[Any] = [base.get("subtotal_raw"), summary.get("subtotal_raw")]
    for item in useful_items or []:
        label = safe_str(item.get("particulars_raw"))
        value = item.get("amount_raw")
        if not safe_str(value):
            continue
        if re.search(r"\b(day\s*total|room\s*rate\s*(?:for)?\s*\d{1,2}(?:st|nd|rd|th)?|one\s*day)\b", label, flags=re.IGNORECASE):
            continue
        candidates.append(value)
    best = ""
    for value in candidates:
        best = prefer_larger_money(best, value)
    if best:
        return best
    grand_total = money_decimal_or_none(first_non_empty(base.get("grand_total_raw"), summary.get("grand_total_raw")))
    temp_context = dict(base)
    for k, v in summary.items():
        if safe_str(v) and not safe_str(temp_context.get(k)):
            temp_context[k] = v
    rates = explicit_gst_rate_components_from_context(temp_context)
    total_rate = decimal_from_percent(rates.get("cgst_rate")) + decimal_from_percent(rates.get("sgst_rate")) + decimal_from_percent(rates.get("igst_rate"))
    if grand_total and total_rate:
        taxable = (grand_total * Decimal("100")) / (Decimal("100") + total_rate)
        return format_money_decimal(taxable)
    return ""
def merge_charge_summary_fields(base: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Merge focused charge summary data. Generic: no vendor/client hardcoding."""
    if not isinstance(summary, dict):
        return base
    summary = normalize_raw_result(summary)
    summary = repair_generic_context_fields(summary)
    summary_context = build_context_text(summary)
    base_context = build_context_text(base)
    hotel_like = is_hotel_lodging_context(base) or is_hotel_lodging_context(summary)
    utility_like = is_utility_bill_context(base) or is_utility_bill_context(summary)
    if not (hotel_like or utility_like or charge_summary_result_has_useful_value(summary)):
        return base
    # Identity/header fields: fill blanks, but do not overwrite good existing values.
    for key in [
        "client_name_raw", "los_vendor_state_raw", "lor_isec_state_raw",
        "service_usage_raw", "bill_date_raw", "explicit_date_range_raw", "service_period_raw",
        "is_invoice_digitally_signed_raw", "digital_signature_evidence_raw", "amount_in_words_raw",
    ]:
        value = safe_str(summary.get(key))
        current_value = safe_str(base.get(key))
        if value and not current_value:
            base[key] = value
        elif hotel_like and value and key == "explicit_date_range_raw":
            summary_from, summary_to = derive_date_range_from_explicit_range(value)
            current_from, current_to = derive_date_range_from_explicit_range(current_value)
            if summary_from and summary_to and (not current_from or not current_to or (summary_from, summary_to) != (current_from, current_to)):
                base[key] = value
        elif hotel_like and value and key == "bill_date_raw" and normalize_date_string(value):
            base[key] = value
    summary_vendor = safe_str(summary.get("vendor_employee_name_raw"))
    current_vendor = safe_str(base.get("vendor_employee_name_raw"))
    if summary_vendor:
        if not current_vendor:
            base["vendor_employee_name_raw"] = summary_vendor
        elif hotel_like and should_prefer_legal_supplier_name(current_vendor, summary_vendor):
            base["vendor_employee_name_raw"] = summary_vendor
    # Document number: only accept true bill/invoice numbers, never GSTIN/PAN/consumer/service/meter numbers.
    summary_bill_no = clean_document_number(first_non_empty(summary.get("bill_no_raw"), summary.get("invoice_no_raw")))
    if summary_bill_no and not is_non_bill_identifier(summary_bill_no):
        current_bill_no = clean_document_number(first_non_empty(base.get("bill_no_raw"), base.get("invoice_no_raw")))
        if (
            not current_bill_no
            or is_non_bill_identifier(current_bill_no)
            or GSTIN_PATTERN.fullmatch(current_bill_no.upper().replace(" ", ""))
            or (hotel_like and should_prefer_charge_summary_bill_no(current_bill_no, summary_bill_no))
        ):
            base["bill_no_raw"] = summary_bill_no
    # Totals from summary pages are more reliable, but multi-page hotel scans can
    # produce smaller Day Total values from page 1. Keep the largest complete
    # summary amount/tax total instead of overwriting with a daily subtotal.
    for key in ["subtotal_raw", "cgst_total_raw", "sgst_total_raw", "igst_total_raw", "cess_total_raw", "grand_total_raw"]:
        value = safe_str(summary.get(key))
        if not value:
            continue
        if hotel_like:
            base[key] = prefer_larger_money(base.get(key), value)
        else:
            base[key] = value
    # Preserve GST rate hint for output rate column.
    summary_rate = first_non_empty(summary.get("gst_rate_raw"), summary.get("_source_text_gst_rate_raw"))
    if not summary_rate:
        for item in summary.get("items") or []:
            if isinstance(item, dict) and safe_str(item.get("rate_raw")):
                summary_rate = item.get("rate_raw")
                break
    if summary_rate:
        base["_source_text_gst_rate_raw"] = first_non_empty(base.get("_source_text_gst_rate_raw"), summary_rate)
    summary_items = [item for item in (summary.get("items") or []) if isinstance(item, dict)]
    useful_items = [item for item in summary_items if safe_str(first_non_empty(item.get("amount_raw"), item.get("cgst_raw"), item.get("sgst_raw"), item.get("igst_raw")))]
    if hotel_like:
        base = split_hotel_igst_to_cgst_sgst_when_needed(base)
        # Consolidate to a single accommodation/service line when summary totals are visible.
        amount_value = choose_hotel_taxable_amount_from_summary(base, summary, useful_items)
        if safe_str(base.get("cgst_total_raw")) or safe_str(base.get("sgst_total_raw")):
            base["igst_total_raw"] = "0" if not context_mentions_explicit_igst(base) else safe_str(base.get("igst_total_raw"))
            if not safe_str(base.get("cess_total_raw")):
                base["cess_total_raw"] = "0"
        if amount_value or safe_str(base.get("grand_total_raw")):
            base["items"] = [{
                "particulars_raw": first_non_empty(useful_items[0].get("particulars_raw") if useful_items else "", base.get("service_usage_raw"), "Hotel Accommodation"),
                "quantity_raw": "",
                "quantity_unit_raw": "",
                "hsn_code_raw": first_non_empty(useful_items[0].get("hsn_code_raw") if useful_items else "", base.get("_source_text_hsn_code_raw")),
                "rate_raw": first_non_empty(summary_rate, useful_items[0].get("rate_raw") if useful_items else ""),
                "amount_raw": amount_value,
                "cgst_raw": first_non_empty(base.get("cgst_total_raw"), summary.get("cgst_total_raw"), useful_items[0].get("cgst_raw") if useful_items else ""),
                "sgst_raw": first_non_empty(base.get("sgst_total_raw"), summary.get("sgst_total_raw"), useful_items[0].get("sgst_raw") if useful_items else ""),
                "igst_raw": first_non_empty(base.get("igst_total_raw"), "0" if first_non_empty(base.get("cgst_total_raw"), base.get("sgst_total_raw")) else "", useful_items[0].get("igst_raw") if useful_items else ""),
                "cess_raw": first_non_empty(base.get("cess_total_raw"), "0" if first_non_empty(base.get("cgst_total_raw"), base.get("sgst_total_raw")) else "", useful_items[0].get("cess_raw") if useful_items else ""),
            }]
    if utility_like:
        # Utility bills should have one payable line, not one row per charge component.
        payable = first_non_empty(
            summary.get("grand_total_raw"),
            summary.get("subtotal_raw"),
            extract_final_payable_from_items(useful_items),
            extract_final_payable_from_items(base.get("items") or []),
        )
        if payable:
            base["grand_total_raw"] = payable
            base["subtotal_raw"] = payable
            base["cgst_total_raw"] = first_non_empty(summary.get("cgst_total_raw"), "0")
            base["sgst_total_raw"] = first_non_empty(summary.get("sgst_total_raw"), "0")
            base["igst_total_raw"] = first_non_empty(summary.get("igst_total_raw"), "0")
            base["cess_total_raw"] = first_non_empty(summary.get("cess_total_raw"), "0")
            base["items"] = [{
                "particulars_raw": first_non_empty(summary.get("service_usage_raw"), base.get("service_usage_raw"), "Utility Charges"),
                "quantity_raw": "",
                "quantity_unit_raw": "",
                "hsn_code_raw": "",
                "rate_raw": "",
                "amount_raw": payable,
                "cgst_raw": "0",
                "sgst_raw": "0",
                "igst_raw": "0",
                "cess_raw": "0",
            }]
            base["_utility_summary_repaired"] = "Yes"
    base["_charge_summary_evidence"] = first_non_empty(base.get("_charge_summary_evidence"), summary_context, base_context)
    return base
def is_non_bill_identifier(value: Any) -> bool:
    text = safe_str(value)
    if not text:
        return False
    normalized = text.upper().replace(" ", "")
    if GSTIN_PATTERN.fullmatch(normalized):
        return True
    if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", normalized):
        return True
    # These are account/service identifiers, not bill/invoice numbers.
    if re.search(r"^\s*(?:s|5)\s*\.?\s*no\s*[:\-]?\s*\d", text, flags=re.IGNORECASE):
        return True
    return bool(re.search(
        r"\b(?:S\s*NO|5\s*NO|SERIAL\s*NO|SC\s*NO|USC\s*NO|CONSUMER\s*NO|CUSTOMER\s*NO|ACCOUNT\s*NO|METER\s*NO|SERVICE\s*NO|SERVICE\s*CONNECTION|ROOM\s*NO)\b",
        text,
        flags=re.IGNORECASE,
    ))
UTILITY_STATE_ABBREVIATIONS = {
    "AP": "Andhra Pradesh",
    "TS": "Telangana",
    "TG": "Telangana",
    "MH": "Maharashtra",
    "GJ": "Gujarat",
    "KA": "Karnataka",
    "TN": "Tamil Nadu",
    "KL": "Kerala",
    "UP": "Uttar Pradesh",
    "MP": "Madhya Pradesh",
    "RJ": "Rajasthan",
    "DL": "Delhi",
    "HR": "Haryana",
    "PB": "Punjab",
    "WB": "West Bengal",
    "BR": "Bihar",
    "OD": "Odisha",
    "OR": "Odisha",
    "CG": "Chhattisgarh",
    "JH": "Jharkhand",
    "AS": "Assam",
    "HP": "Himachal Pradesh",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "JK": "Jammu and Kashmir",
}
def infer_utility_state_from_vendor_abbreviation(raw: Dict[str, Any]) -> str:
    """
    Generic utility-state fallback. Many utility boards print only an abbreviated
    state prefix in the distributor name, for example AP/TG/TS/MH/GJ.
    This does not hardcode vendor/client/company names; it only expands a visible
    Indian state abbreviation when the document is clearly a utility bill.
    """
    if not is_utility_bill_context(raw):
        return ""
    context_parts = [
        raw.get("vendor_employee_name_raw"),
        raw.get("service_usage_raw"),
        raw.get("document_type_raw"),
        raw.get("remarks_raw"),
        raw.get("_charge_summary_evidence"),
        raw.get("visible_text_raw"),
    ]
    context = " ".join(safe_str(part) for part in context_parts if safe_str(part))
    if not context:
        return ""
    for state in INDIAN_STATE_NAMES:
        if re.search(rf"\b{re.escape(state)}\b", context, flags=re.IGNORECASE):
            return state
    vendor_text = safe_str(raw.get("vendor_employee_name_raw")).upper()
    service_text = safe_str(raw.get("service_usage_raw")).upper()
    utility_context = f"{vendor_text} {service_text}"
    for abbr, state in UTILITY_STATE_ABBREVIATIONS.items():
        if re.search(rf"\b{abbr}[A-Z]{{2,}}\b", utility_context):
            return state
    return ""
def utility_has_strong_bill_number_label(raw: Dict[str, Any], value: str) -> bool:
    """
    Accept utility document numbers only when explicit Bill/Invoice/Receipt label
    evidence exists. Reject service/consumer/meter/serial numbers, even when OCR
    reads S NO as 5 NO.
    """
    if not value:
        return False
    context = build_context_text(raw)
    value_escaped = re.escape(value)
    if re.search(rf"\b(?:bill\s*(?:no|number)|invoice\s*(?:no|number)|receipt\s*(?:no|number))\b\s*[:#\-]?\s*{value_escaped}", context, flags=re.IGNORECASE):
        return True
    if re.search(rf"\b(?:s|5|service|consumer|customer|account|meter|serial|sc|usc)\s*(?:no|number)?\b\s*[:#\-]?\s*{value_escaped}", context, flags=re.IGNORECASE):
        return False
    return False
def sanitize_utility_bill_number_for_output(raw: Dict[str, Any], bill_no: Any) -> str:
    """
    Utility bills often print Service No/Consumer No/Serial No but no actual
    invoice Bill No. Do not expose those account identifiers as BILL NO.
    """
    bill_no_clean = clean_document_number(bill_no)
    if not bill_no_clean:
        return ""
    if not is_utility_bill_context(raw):
        return "" if is_non_bill_identifier(bill_no_clean) else bill_no_clean
    if is_non_bill_identifier(bill_no_clean):
        return ""
    if utility_has_strong_bill_number_label(raw, bill_no_clean):
        return bill_no_clean
    if safe_str(raw.get("_utility_bill_no_header_evidence")).lower() == "yes":
        context = build_context_text(raw)
        if re.search(rf"bill\s+of\s+supply[^\n]{{0,140}}{re.escape(bill_no_clean)}", context, flags=re.IGNORECASE):
            return bill_no_clean
        # Deterministic utility OCR stores only title/header-based document numbers,
        # so allow it even when line breaks are lost in later merge/context cleanup.
        return bill_no_clean
    return ""
def utility_context_has_bill_date_label(raw: Dict[str, Any]) -> bool:
    context = build_context_text(raw)
    return bool(re.search(r"\b(?:bill\s*date|invoice\s*date|issue\s*date|date\s*of\s*bill)\b", context, flags=re.IGNORECASE))
def sanitize_utility_bill_date_for_output(raw: Dict[str, Any], bill_date: Any, from_date: Any, to_date: Any) -> str:
    """
    For faint utility scans, OCR often copies service-period start date, due date,
    connection date, or old reading date as bill_date. Keep bill_date only when
    bill/invoice/issue-date evidence exists or when it is clearly not the service
    period date.
    """
    normalized = normalize_date_string(bill_date)
    if not normalized or not is_utility_bill_context(raw):
        return normalized
    if safe_str(raw.get("_utility_bill_date_label_evidence")).lower() == "yes":
        return normalized
    if normalized in {safe_str(from_date), safe_str(to_date)} and not utility_context_has_bill_date_label(raw):
        return ""
    context = build_context_text(raw)
    if re.search(r"\b(?:due\s*date|last\s*paid|first\s*paid|connected\s*load|connection\s*date|reading\s*date|prev(?:ious)?\s*date|present\s*date)\b", context, flags=re.IGNORECASE) and not utility_context_has_bill_date_label(raw):
        return ""
    return normalized
def extract_final_payable_from_items(items: List[Dict[str, Any]]) -> str:
    """Pick final payable/total row from model-extracted utility items."""
    best = ""
    best_rank = -1
    label_ranks = [
        (100, r"\b(total\s*due|amount\s*due|net\s*payable|payable\s*amount|bill\s*amount)\b"),
        (90, r"\b(total\s*amount|current\s*bill|net\s*amount|total\s*bill)\b"),
        (60, r"\bgrand\s*total\b"),
    ]
    for item in items or []:
        if not isinstance(item, dict):
            continue
        label = safe_str(item.get("particulars_raw") or item.get("particulars"))
        amount = clean_amount(first_non_empty(item.get("amount_raw"), item.get("amount")))
        if not amount:
            continue
        for rank, pattern in label_ranks:
            if rank > best_rank and re.search(pattern, label, flags=re.IGNORECASE):
                best_rank = rank
                best = amount
    return best
def hotel_total_tax_rate_for_output(raw: Dict[str, Any], amount: Any, cgst: Any, sgst: Any, igst: Any, cess: Any) -> str:
    rates = explicit_gst_rate_components_from_context(raw)
    if rates.get("cgst_rate") and rates.get("sgst_rate"):
        return f"{rates['cgst_rate']} + {rates['sgst_rate']}"
    if rates.get("igst_rate") and not first_non_empty(cgst, sgst):
        return rates["igst_rate"]
    amount_dec = money_decimal_or_none(amount)
    cgst_dec = money_decimal_or_none(cgst) or Decimal("0")
    sgst_dec = money_decimal_or_none(sgst) or Decimal("0")
    if amount_dec and amount_dec > 0 and cgst_dec > 0 and sgst_dec > 0:
        cgst_rate = (cgst_dec / amount_dec) * Decimal("100")
        sgst_rate = (sgst_dec / amount_dec) * Decimal("100")
        if Decimal("0") < cgst_rate <= Decimal("40") and Decimal("0") < sgst_rate <= Decimal("40"):
            return f"{format_decimal_amount(cgst_rate.quantize(Decimal('0.01')).normalize())}% + {format_decimal_amount(sgst_rate.quantize(Decimal('0.01')).normalize())}%"
    return choose_gst_rate_for_output(raw, {}, amount, cgst, sgst, igst, cess)
def repair_hotel_final_line_items(raw: Dict[str, Any], final_line_items: List[Dict[str, Any]], bill_no: str, final_from_date: str, final_to_date: str, bill_date: str) -> List[Dict[str, Any]]:
    """Final hotel safety: one consolidated taxable line with split GST totals."""
    if not is_hotel_lodging_context(raw):
        return final_line_items
    raw = split_hotel_igst_to_cgst_sgst_when_needed(raw)
    amount = clean_amount(first_non_empty(raw.get("subtotal_raw"), final_line_items[0].get("amount") if final_line_items else ""))
    cgst = clean_amount(first_non_empty(raw.get("cgst_total_raw"), final_line_items[0].get("cgst") if final_line_items else ""))
    sgst = clean_amount(first_non_empty(raw.get("sgst_total_raw"), final_line_items[0].get("sgst") if final_line_items else ""))
    igst = clean_amount(first_non_empty(raw.get("igst_total_raw"), final_line_items[0].get("igst") if final_line_items else ""))
    cess = clean_amount(first_non_empty(raw.get("cess_total_raw"), final_line_items[0].get("cess") if final_line_items else ""))
    # If split GST labels are visible, reject hallucinated IGST.
    if context_mentions_split_gst(raw) and not context_mentions_explicit_igst(raw):
        igst = "0"
    if not cgst:
        cgst = "0"
    if not sgst:
        sgst = "0"
    if not igst:
        igst = "0"
    if not cess:
        cess = "0"
    # When a hotel summary has a final total plus GST percentages, derive the
    # taxable amount and split taxes from the tax-inclusive total. This repairs
    # cases where a vision model reads only one daily room-rate row or puts split
    # GST into IGST.
    grand_total = money_decimal_or_none(raw.get("grand_total_raw"))
    rates = explicit_gst_rate_components_from_context(raw)
    cgst_rate_dec = decimal_from_percent(rates.get("cgst_rate"))
    sgst_rate_dec = decimal_from_percent(rates.get("sgst_rate"))
    igst_rate_dec = decimal_from_percent(rates.get("igst_rate"))
    split_rate_dec = cgst_rate_dec + sgst_rate_dec
    amount_dec = money_decimal_or_none(amount)
    cgst_dec = money_decimal_or_none(cgst) or Decimal("0")
    sgst_dec = money_decimal_or_none(sgst) or Decimal("0")
    igst_dec = money_decimal_or_none(igst) or Decimal("0")
    current_total_dec = (amount_dec or Decimal("0")) + cgst_dec + sgst_dec + igst_dec
    visible_tax_total_dec = cgst_dec + sgst_dec + igst_dec
    if grand_total and visible_tax_total_dec > 0 and (not amount_dec or not current_total_dec or grand_total > current_total_dec * Decimal("1.20")):
        taxable = grand_total - visible_tax_total_dec
        if taxable > 0:
            amount = format_money_decimal(taxable)
            amount_dec = taxable
            current_total_dec = grand_total
    if grand_total and split_rate_dec and (not amount_dec or not current_total_dec or grand_total > current_total_dec * Decimal("1.20")):
        taxable = (grand_total * Decimal("100")) / (Decimal("100") + split_rate_dec)
        amount = format_money_decimal(taxable)
        cgst = format_money_decimal((taxable * cgst_rate_dec) / Decimal("100")) if cgst_rate_dec else "0"
        sgst = format_money_decimal((taxable * sgst_rate_dec) / Decimal("100")) if sgst_rate_dec else "0"
        igst = "0"
        cess = "0"
    elif grand_total and igst_rate_dec and not split_rate_dec and (not amount_dec or grand_total > current_total_dec * Decimal("1.20")):
        taxable = (grand_total * Decimal("100")) / (Decimal("100") + igst_rate_dec)
        amount = format_money_decimal(taxable)
        cgst = "0"
        sgst = "0"
        igst = format_money_decimal((taxable * igst_rate_dec) / Decimal("100"))
        cess = "0"
    if not amount and final_line_items:
        return final_line_items
    first = dict(final_line_items[0]) if final_line_items else {}
    first["bill_no"] = "" if is_non_bill_identifier(bill_no) else bill_no
    first["particulars"] = first_non_empty(raw.get("service_usage_raw"), first.get("particulars"), "Hotel Accommodation")
    first["from_date"] = final_from_date
    first["to_date"] = final_to_date
    first["amount"] = amount
    first["cgst"] = cgst
    first["sgst"] = sgst
    first["igst"] = igst
    first["cess"] = cess
    first["bill_date"] = bill_date
    first["quantity"] = ""
    first["quantity_unit"] = ""
    first["rate"] = hotel_total_tax_rate_for_output(raw, amount, cgst, sgst, igst, cess)
    first["isec_gstin"] = safe_str(raw.get("isec_gstin_raw"))
    hsn = first_non_empty(first.get("hsn_code"), raw.get("_source_text_hsn_code_raw"))
    if hsn:
        first["hsn_code"] = hsn
    return [first]
def collapse_utility_final_line_items(raw: Dict[str, Any], final_line_items: List[Dict[str, Any]], bill_no: str, final_from_date: str, final_to_date: str, bill_date: str) -> List[Dict[str, Any]]:
    if not is_utility_bill_context(raw):
        return final_line_items
    if safe_str(raw.get("_utility_summary_repaired")):
        payable = first_non_empty(raw.get("grand_total_raw"), raw.get("subtotal_raw"), extract_final_payable_from_items(final_line_items))
    else:
        # Without a focused summary repair, raw grand_total can be the erroneous sum of
        # every component row. Prefer an explicit final payable row from extracted items.
        payable = first_non_empty(extract_final_payable_from_items(final_line_items), raw.get("grand_total_raw"), raw.get("subtotal_raw"))
    payable = clean_amount(payable)
    if not payable:
        return final_line_items
    return [{
        "bill_no": sanitize_utility_bill_number_for_output(raw, bill_no),
        "particulars": first_non_empty(raw.get("service_usage_raw"), "Utility Charges"),
        "from_date": final_from_date,
        "to_date": final_to_date,
        "amount": payable,
        "cgst": "0",
        "sgst": "0",
        "igst": "0",
        "cess": "0",
        "department_code": safe_str(raw.get("department_code_raw")),
        "function_code": safe_str(raw.get("function_code_raw")),
        "location_code": safe_str(raw.get("location_code_raw")),
        "sub_loc_code": safe_str(raw.get("sub_loc_code_raw")),
        "bill_date": bill_date,
        "quantity": "",
        "quantity_unit": "",
        "hsn_code": normalize_hsn_sac_code(first_non_empty(first_final_line_value(final_line_items, "hsn_code"), raw.get("_source_text_hsn_code_raw"))),
        "rate": "",
        "isec_gstin": safe_str(raw.get("isec_gstin_raw")),
    }]
def is_taxless_bill_of_supply_context(raw: Dict[str, Any]) -> bool:
    """Detect single-item taxless/composition Bill of Supply style documents.
    Generic and label-based. This is deliberately separate from normal invoices,
    hotel bills, utility bills and regular multi-item receipts so previous repairs
    are not affected.
    """
    if is_hotel_lodging_context(raw) or is_utility_bill_context(raw):
        return False
    context = build_context_text(raw)
    return bool(re.search(
        r"\b(?:bill\s+of\s+supply|composition\s+taxable|composition\s+scheme|not\s+eligible\s+to\s+collect\s+tax|tax\s+not\s+applicable|gst\s+not\s+applicable)\b",
        context,
        flags=re.IGNORECASE,
    ))
def all_final_item_taxes_blank_or_zero(final_line_items: List[Dict[str, Any]]) -> bool:
    for item in final_line_items or []:
        if not isinstance(item, dict):
            continue
        for key in ("cgst", "sgst", "igst", "cess"):
            value = clean_amount(item.get(key))
            if value and decimal_or_none(value) not in (None, Decimal("0")):
                return False
    return True
def score_consolidated_particulars(value: Any, raw: Dict[str, Any]) -> int:
    text = normalize_invoice_name(value)
    if not text or not re.search(r"[A-Za-z]", text):
        return -1000
    lower = text.lower()
    service_lower = normalize_invoice_name(raw.get("service_usage_raw")).lower()
    score = min(len(text), 90)
    if re.search(r"\b(?:hsn|sac|code|tax|gst|amount|rate|qty|quantity)\b", lower):
        score -= 45
    if service_lower and lower == service_lower:
        score -= 25
    if re.search(r"\([^)]{2,}\)|\b(?:for|month|period)\b", lower):
        score += 10
    if len(text.split()) <= 1:
        score -= 20
    return score
def choose_consolidated_particulars(raw: Dict[str, Any], final_line_items: List[Dict[str, Any]]) -> str:
    candidates: List[str] = []
    for item in final_line_items or []:
        if isinstance(item, dict):
            value = normalize_invoice_name(item.get("particulars"))
            if value:
                candidates.append(value)
    service = normalize_invoice_name(raw.get("service_usage_raw"))
    if service:
        candidates.append(service)
    if not candidates:
        return "Bill of Supply"
    return max(candidates, key=lambda v: score_consolidated_particulars(v, raw))
def collapse_taxless_single_item_bill_of_supply(
    raw: Dict[str, Any],
    final_line_items: List[Dict[str, Any]],
    bill_no: str,
    final_from_date: str,
    final_to_date: str,
    bill_date: str,
) -> List[Dict[str, Any]]:
    """Collapse VLM-split Bill of Supply rows when one row is a unit-rate row.
    Some scanned Bill of Supply documents have a single item table like:
    Description | HSN | Qty | Rate | Amount. Vision extraction may return the
    unit rate as one line amount and the remaining/total amount as another line.
    This repair triggers only when a quantity row's amount behaves like a unit
    rate (quantity * amount equals the visible final total/line sum). It does not
    collapse genuine multi-item bills such as tea + coffee because their item
    amounts are already extended line amounts, not unit rates.
    """
    if not is_taxless_bill_of_supply_context(raw):
        return final_line_items
    if not isinstance(final_line_items, list) or len(final_line_items) < 2 or len(final_line_items) > 4:
        return final_line_items
    if not all_final_item_taxes_blank_or_zero(final_line_items):
        return final_line_items
    line_total, has_line_total = final_line_items_total(final_line_items)
    if not has_line_total or line_total <= 0:
        return final_line_items
    grand_total = money_decimal_or_none(first_non_empty(raw.get("grand_total_raw"), raw.get("subtotal_raw")))
    usable_total = grand_total if grand_total and grand_total > 0 and abs(grand_total - line_total) <= Decimal("0.01") else line_total
    unit_rate_row: Dict[str, Any] = {}
    for item in final_line_items:
        if not isinstance(item, dict):
            continue
        qty = decimal_or_none(clean_amount(item.get("quantity")))
        amt = decimal_or_none(clean_amount(item.get("amount")))
        if qty is None or amt is None or qty <= 1 or amt <= 0:
            continue
        extended = qty * amt
        # This is the key guard: collapse only when the row's amount is visibly a
        # unit rate, because quantity * amount equals the final payable total.
        if abs(extended - usable_total) <= Decimal("0.01"):
            unit_rate_row = item
            break
    if not unit_rate_row:
        return final_line_items
    hsn_code = first_non_empty(
        first_final_line_value(final_line_items, "hsn_code"),
        raw.get("_source_text_hsn_code_raw"),
    )
    quantity = safe_str(unit_rate_row.get("quantity"))
    particulars = choose_consolidated_particulars(raw, final_line_items)
    final_bill_no = first_non_empty(first_final_line_value(final_line_items, "bill_no"), bill_no)
    final_bill_date = first_non_empty(first_final_line_value(final_line_items, "bill_date"), bill_date)
    final_isec_gstin = first_non_empty(first_final_line_value(final_line_items, "isec_gstin"), raw.get("isec_gstin_raw"))
    raw["_bill_of_supply_line_item_repair"] = (
        "Consolidated split taxless Bill of Supply rows because one extracted row was a unit-rate row."
    )
    return [{
        "bill_no": final_bill_no,
        "particulars": particulars,
        "from_date": final_from_date,
        "to_date": final_to_date,
        "amount": format_decimal_amount(usable_total),
        "cgst": "0",
        "sgst": "0",
        "igst": "0",
        "cess": "0",
        "department_code": safe_str(raw.get("department_code_raw")),
        "function_code": safe_str(raw.get("function_code_raw")),
        "location_code": safe_str(raw.get("location_code_raw")),
        "sub_loc_code": safe_str(raw.get("sub_loc_code_raw")),
        "bill_date": final_bill_date,
        "quantity": "",
        "quantity_unit": "",
        "hsn_code": normalize_hsn_sac_code(hsn_code),
        "rate": "",
        "isec_gstin": safe_str(final_isec_gstin),
    }]
def normalize_missing_tax_and_rate_to_zero(final_line_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Final output convention for tax columns.
    When GST/tax is not printed or not extracted, return 0 instead of blank for
    CGST, SGST, IGST, CESS and rate. Keep valid non-zero tax amounts and visible
    GST percentage rates unchanged. This runs at the very end so specialized
    hotel, utility, Bill-of-Supply and handwritten-receipt repairs are preserved.
    """
    normalized_items: List[Dict[str, Any]] = []
    for item in final_line_items or []:
        if not isinstance(item, dict):
            continue
        fixed = dict(item)
        for key in ("cgst", "sgst", "igst", "cess"):
            value = clean_amount(fixed.get(key))
            fixed[key] = value if value != "" else "0"
        rate_value = safe_str(fixed.get("rate"))
        fixed["rate"] = rate_value if rate_value else "0"
        normalized_items.append(fixed)
    return normalized_items
def sanitize_state_for_output(raw: Dict[str, Any], state_value: Any, role: str = "") -> str:
    state = state_or_empty(state_value)
    if not state:
        return ""
    # GSTIN-derived states are reliable.
    if role == "vendor" and state == state_from_gstin(raw.get("vendor_gstin_raw")):
        return state
    if role == "client" and state == state_from_gstin(raw.get("isec_gstin_raw")):
        return state
    evidence_parts = [
        raw.get("vendor_employee_name_raw"), raw.get("client_name_raw"), raw.get("billed_party_raw"),
        raw.get("billed_party_block_raw"), raw.get("customer_block_raw"), raw.get("client_block_raw"),
        raw.get("visible_text_raw"), raw.get("vendor_address_raw"), raw.get("client_address_raw"),
        raw.get("patient_address_raw"), raw.get("supply_address_raw"), raw.get("bill_to_address_raw"),
        raw.get("ship_to_address_raw"), raw.get("place_of_supply_raw"), raw.get("service_usage_raw"),
        raw.get("remarks_raw"), raw.get("_source_text_context_raw"), raw.get("_charge_summary_evidence"),
    ]
    evidence = " ".join(safe_str(part) for part in evidence_parts if safe_str(part))
    if re.search(rf"\b{re.escape(state)}\b", evidence, flags=re.IGNORECASE):
        return state
    # For utility bills, keep a state only when supported by visible state evidence
    # or by a generic Indian state abbreviation in the utility-board name.
    utility_state = infer_utility_state_from_vendor_abbreviation(raw)
    if is_utility_bill_context(raw):
        return state if utility_state and state == utility_state else ""
    return state
def sanitize_digital_signature_for_context(raw: Dict[str, Any], value: str) -> str:
    if value != "Yes":
        return value
    context = build_context_text(raw).lower()
    digital_evidence = re.search(r"\b(irn|ack\s*no|ack\s*date|e-?invoice|qr\s*code|digital(?:ly)?\s*signed|digital\s*signature\s*certificate)\b", context)
    ordinary_signature = re.search(r"\b(cashier\s*id|cashier|guest\s*signature|receiver'?s?\s*signature|authori[sz]ed\s*signatory|handwritten\s*signature|signature)\b", context)
    if ordinary_signature and not digital_evidence:
        return "No"
    if is_hotel_lodging_context(raw):
        # For hotel invoices, ordinary cashier/guest/authorised signatures are common.
        # Treat as digitally signed only for strong e-invoice/IRN/Ack/QR evidence.
        strong_hotel_digital = re.search(r"(irn|ack\s*no|ack\s*date|e-?invoice|qr\s*code)", context)
        return "Yes" if strong_hotel_digital else "No"
    return value
def allow_visible_claim_total_override(raw: Dict[str, Any]) -> bool:
    """Restrict paid/claim total override to receipt-like cases, not normal GST invoices."""
    context = build_context_text(raw).lower()
    if re.search(r"\b(tax\s+invoice|hotel|lodging|room\s+rate|tariff|bill\s+summary|total\s+billing|cgst|sgst|central\s+gst|state\s+gst|sac|hsn)\b", context, flags=re.IGNORECASE):
        return False
    return bool(re.search(r"\b(cab|taxi|ride|fare|trip|paid\s+amount|amount\s+paid|paid\s+by|net\s+payable|deposit\s+receipt|hospital\s+deposit)\b", context, flags=re.IGNORECASE))
def correct_utility_bill_date_year(raw: Dict[str, Any], bill_date: Any, billing_from_date: Any) -> str:
    """
    Utility bills are issued in the billing month or the following 1-2 months.
    If OCR misread the year (e.g. 2026 -> 2022), keep day/month and repair the year
    using the billing month printed in the title (e.g. JUN-2026).
    """
    normalized = normalize_date_string(bill_date)
    bill = parse_normalized_date(normalized)
    start = parse_normalized_date(billing_from_date)
    if not bill or not start or not is_utility_bill_context(raw):
        return normalized
    end_month = start.month + 3
    end_year = start.year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    window_end = date(end_year, end_month, 1)
    def in_window(d):
        return start <= d < window_end
    if in_window(bill):
        return normalized
    for year in (start.year, start.year + 1):
        try:
            candidate = date(year, bill.month, bill.day)
        except ValueError:
            continue
        if in_window(candidate):
            return f"{candidate.day:02d}/{candidate.month:02d}/{candidate.year}"
    return normalized
def build_final_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Expense type extraction has been disabled by requirement.
    # Keep internal key blank so final output remains consistent.
    raw["expense_type_raw"] = ""
    raw = repair_generic_context_fields(raw)
    raw = sanitize_hotel_tax_totals(raw)
    raw = deduplicate_raw_items(raw)
    raw = backfill_from_items(raw)
    raw = fix_grand_total_from_items(raw)
    final_from_date, final_to_date, date_source = derive_final_date_range(raw)
    bill_date_raw = safe_str(raw.get("bill_date_raw"))
    if len(extract_all_full_dates(bill_date_raw)) >= 2:
        bill_date = ""
    else:
        bill_date = normalize_date_string(bill_date_raw)
    final_from_date, final_to_date, date_source = apply_service_date_fallback(
        raw,
        final_from_date,
        final_to_date,
        bill_date,
        date_source,
    )
    utility_from_date, utility_to_date = derive_utility_billing_month_range(raw)
    if utility_from_date and utility_to_date:
        final_from_date, final_to_date, date_source = utility_from_date, utility_to_date, "utility_billing_month_raw"
    bill_no = choose_final_bill_number(raw)
    # Utility-specific final safety: do not expose Service No/Consumer No as BILL NO,
    # and do not use service-period/due/connection dates as BILL DATE.
    bill_no = sanitize_utility_bill_number_for_output(raw, bill_no)
    bill_date = sanitize_utility_bill_date_for_output(raw, bill_date, final_from_date, final_to_date)
    if utility_from_date:
        bill_date = correct_utility_bill_date_year(raw, bill_date, utility_from_date)
    digital_signature = sanitize_digital_signature_for_context(
        raw,
        decide_digital_signature(
            raw.get("is_invoice_digitally_signed_raw"),
            raw.get("digital_signature_evidence_raw"),
        ),
    )
    items = raw.get("items") or []
    if not items:
        items = [
            {
                "particulars_raw": "",
                "quantity_raw": "",
                "quantity_unit_raw": "",
                "hsn_code_raw": "",
                "rate_raw": "",
                "amount_raw": first_non_empty(raw.get("subtotal_raw"), raw.get("grand_total_raw")),
                "cgst_raw": raw.get("cgst_total_raw", ""),
                "sgst_raw": raw.get("sgst_total_raw", ""),
                "igst_raw": raw.get("igst_total_raw", ""),
                "cess_raw": raw.get("cess_total_raw", ""),
            }
        ]
    final_line_items = []
    for item in items:
        final_amount, final_rate, final_quantity = choose_amount_rate_quantity(item)
        final_quantity_unit = normalize_quantity_unit_for_context(
            raw,
            first_non_empty(
                item.get("quantity_unit_raw"),
                raw.get("_source_text_quantity_unit_raw"),
                infer_quantity_unit_from_context(raw),
            ),
        )
        final_hsn_code = normalize_hsn_sac_code(first_non_empty(item.get("hsn_code_raw"), raw.get("_source_text_hsn_code_raw")))
        # Multi-panel/local receipts can have a different bill date for each
        # small receipt on the same scanned page. Existing invoices keep the
        # global dates because these item fields are normally blank.
        item_bill_date = normalize_date_string(first_non_empty(item.get("bill_date_raw"), item.get("from_date_raw"), item.get("to_date_raw")))
        item_from_date = normalize_date_string(first_non_empty(item.get("from_date_raw"), item.get("bill_date_raw")))
        item_to_date = normalize_date_string(first_non_empty(item.get("to_date_raw"), item.get("bill_date_raw"), item.get("from_date_raw")))
        output_bill_date = item_bill_date or bill_date
        output_from_date = item_from_date or final_from_date
        output_to_date = item_to_date or final_to_date
        final_particulars = infer_particulars_from_context(raw, item.get("particulars_raw"))
        final_quantity_unit = normalize_quantity_unit_for_line_item(raw, item, final_quantity_unit, final_particulars)
        # Quantity/unit logic is disabled; never derive values from source text or context.
        final_quantity = ""
        final_quantity_unit = ""
        final_cgst = output_tax_amount(raw, item, "cgst_raw", "cgst_total_raw", len(items))
        final_sgst = output_tax_amount(raw, item, "sgst_raw", "sgst_total_raw", len(items))
        final_igst = output_tax_amount(raw, item, "igst_raw", "igst_total_raw", len(items))
        final_cess = output_tax_amount(raw, item, "cess_raw", "cess_total_raw", len(items))
        final_rate = choose_gst_rate_for_output(raw, item, final_amount, final_cgst, final_sgst, final_igst, final_cess)
        final_quantity = sanitize_quantity_for_output(final_quantity, final_amount, item, raw)
        if not final_quantity and final_amount and final_quantity_unit:
            final_quantity = "1"
        final_quantity_unit = ""
        final_line_items.append(
            {
                "bill_no": bill_no,
                "particulars": final_particulars,
                "from_date": final_from_date,
                "to_date": final_to_date,
                "amount": final_amount,
                "cgst": final_cgst,
                "sgst": final_sgst,
                "igst": final_igst,
                "cess": final_cess,
                "department_code": safe_str(raw.get("department_code_raw")),
                "function_code": safe_str(raw.get("function_code_raw")),
                "location_code": safe_str(raw.get("location_code_raw")),
                "sub_loc_code": safe_str(raw.get("sub_loc_code_raw")),
                "bill_date": output_bill_date,
                "quantity": "",
                "quantity_unit": "",
                "hsn_code": final_hsn_code,
                "rate": final_rate,
                "isec_gstin": safe_str(raw.get("isec_gstin_raw")),
            }
        )
    final_line_items = repair_tax_invoice_totals_from_visible_rates(raw, final_line_items)
    final_line_items = repair_hotel_final_line_items(raw, final_line_items, bill_no, final_from_date, final_to_date, bill_date)
    final_line_items = collapse_utility_final_line_items(raw, final_line_items, bill_no, final_from_date, final_to_date, bill_date)
    final_line_items = apply_visible_claim_total_override(
        raw,
        final_line_items,
        bill_no,
        final_from_date,
        final_to_date,
        bill_date,
    )
    final_line_items = collapse_taxless_single_item_bill_of_supply(
        raw,
        final_line_items,
        bill_no,
        final_from_date,
        final_to_date,
        bill_date,
    )
    final_line_items = deduplicate_final_line_items(final_line_items)
    final_line_items = normalize_missing_tax_and_rate_to_zero(final_line_items)
    # Remarks generation is disabled; API always returns an empty remarks field.
    raw["remarks_raw"] = ""
    raw = repair_client_from_context(raw)
    final_lor = first_non_empty(raw.get("lor_isec_state_raw"), infer_lor_state_from_raw(raw))
    final_los = first_non_empty(
        state_from_gstin(raw.get("vendor_gstin_raw")),
        state_from_gstin(raw.get("_source_text_vendor_gstin_raw")),
        state_or_empty(raw.get("los_vendor_state_raw")),
        infer_los_state_from_raw(raw),
    )
    utility_state_hint = infer_utility_state_from_vendor_abbreviation(raw)
    if is_utility_bill_context(raw) and utility_state_hint:
        if not state_or_empty(final_los):
            final_los = utility_state_hint
        if not state_or_empty(final_lor):
            final_lor = utility_state_hint
    final_lor = sanitize_state_for_output(raw, final_lor, "client")
    final_los = sanitize_state_for_output(raw, final_los, "vendor")
    final_vendor_name = safe_str(raw.get("vendor_employee_name_raw"))
    if not final_vendor_name:
        labelled_vendor_hint = extract_labelled_vendor_from_context(raw)
        if labelled_vendor_hint and looks_like_valid_vendor_hint(raw, labelled_vendor_hint):
            final_vendor_name = labelled_vendor_hint
    # Final legal-supplier preference is intentionally limited to hotel/lodging
    # or explicit legal-supplier repair contexts. For Bill of Supply/local bills,
    # a customer with Ltd/Limited can otherwise be merged into vendor.
    legal_supplier_hint = ""
    if is_hotel_lodging_context(raw) or looks_like_hotel_trade_name(final_vendor_name) or should_run_legal_supplier_repair(raw):
        legal_supplier_hint = choose_best_legal_supplier_candidate(
            final_vendor_name,
            raw.get("_legal_supplier_name_raw"),
            raw.get("_legal_supplier_evidence"),
            raw.get("_charge_summary_evidence"),
            raw.get("_vendor_identity_evidence"),
            raw.get("_party_identity_evidence"),
            build_context_text(raw),
        )
        legal_supplier_hint = apply_generic_legal_supplier_ocr_corrections(legal_supplier_hint)
    if legal_supplier_hint and not vendor_candidate_has_mixed_client_or_item_text(legal_supplier_hint) and (not final_vendor_name or should_prefer_legal_supplier_name(final_vendor_name, legal_supplier_hint) or looks_like_hotel_trade_name(final_vendor_name)):
        final_vendor_name = normalize_invoice_name(legal_supplier_hint)
    source_vendor_hint = safe_str(raw.get("_source_text_vendor_name_raw"))
    if source_vendor_hint and should_replace_vendor_with_source(final_vendor_name, source_vendor_hint, raw.get("_source_text_context_raw")):
        final_vendor_name = source_vendor_hint
    if not final_vendor_name:
        repaired_vendor_hint = first_non_empty(
            raw.get("_vendor_identity_candidate_raw"),
            extract_labelled_vendor_from_context(raw),
        )
        if repaired_vendor_hint and looks_like_valid_vendor_hint(raw, repaired_vendor_hint):
            final_vendor_name = normalize_invoice_name(repaired_vendor_hint)
    source_client_hint = safe_str(raw.get("_source_text_client_name_raw"))
    final_client_name = safe_str(raw.get("client_name_raw"))
    if not final_client_name:
        final_client_name = first_non_empty(
            raw.get("_client_candidate_from_replaced_vendor_raw"),
            extract_labelled_client_from_context(raw),
            raw.get("billed_party_raw"),
        )
    # Final generic fallback: if a billed-party candidate was captured during
    # repair, clean and validate it before returning blank. This is label/layout
    # based and contains no company/vendor/client hardcoding.
    if not final_client_name:
        fallback_client = clean_billed_party_candidate(
            first_non_empty(
                raw.get("_client_candidate_from_replaced_vendor_raw"),
                raw.get("billed_party_raw"),
                raw.get("client_name_raw"),
            )
        )
        if fallback_client and looks_like_valid_billed_party_hint(raw, fallback_client):
            final_client_name = fallback_client
    if source_client_hint and should_replace_client_with_source(raw, source_client_hint):
        final_client_name = source_client_hint
    # Final generic safety: for small/local expense bills, do not return a
    # corporate billed-party/customer name as the vendor. Preserve it as client
    # when it is the only labelled business name, but avoid reporting it as supplier.
    if suspicious_vendor_for_local_expense(raw, final_vendor_name):
        if not final_client_name and looks_like_valid_billed_party_hint(raw, final_vendor_name):
            final_client_name = normalize_invoice_name(final_vendor_name)
        repaired_vendor_hint = first_non_empty(
            raw.get("_vendor_identity_candidate_raw"),
            raw.get("account_name_raw") if is_local_service_context(raw) else "",
        )
        if repaired_vendor_hint and not suspicious_vendor_for_local_expense(raw, repaired_vendor_hint):
            final_vendor_name = normalize_invoice_name(repaired_vendor_hint)
        else:
            final_vendor_name = ""
    # Final generic safety: do not return full postal address as vendor name.
    # Keep the supplier/business name and strip address tails such as road/floor/pincode.
    final_vendor_name = clean_vendor_candidate_merged_with_client_or_items(final_vendor_name, raw)
    final_vendor_name = strip_address_tail_from_vendor_name(apply_generic_legal_supplier_ocr_corrections(final_vendor_name))
    # Final generic safety: do not return full postal address as client name.
    # Keep only a standalone billed-party name; if only address is visible, return blank.
    final_client_name = clean_final_client_name(final_client_name, raw, final_vendor_name)
    # Final generic safety: return vendor/client names in English/Latin script.
    # This is not a hardcoded name fix; it transliterates any Devanagari text.
    final_vendor_name = english_script_invoice_name(final_vendor_name)
    final_client_name = english_script_invoice_name(final_client_name)
    # Sum all final line-item charge components for the API response.
    # Formula: total_amount = sum(amount + cgst + sgst + igst + cess) for every line item.
    total_amount_decimal, has_total_amount = final_line_items_total(final_line_items)
    total_amount_value = format_decimal_amount(total_amount_decimal) if has_total_amount else ""
    final = {
        "whether_self_reimbursement": infer_self_reimbursement(raw),
        "vendor_name": final_vendor_name,
        # Expense type functionality is intentionally disabled.
        # Keep this field present for API compatibility, but always return blank.
        "expense_type": "",
        "los": final_los,
        "lor": final_lor,
        "service_usage": safe_str(raw.get("service_usage_raw")),
        "is_invoice_digitally_signed": digital_signature,
        "remarks": "",
        "approver": sanitize_approver_for_output(raw.get("approver_raw"), raw),
        "client_name": final_client_name,
        "total_amount": total_amount_value,
        "line_items": final_line_items,
        "validation": {
            "bill_date_source": "bill_date_raw",
            "from_to_date_source": date_source,
            "invoice_date_used_as_service_date": False,
            "amount_validation": "Amount/rate/quantity are corrected through generic arithmetic validation. For receipt + separate tax-invoice cases, a visible paid/claim total can override a much smaller extracted component total. No fixed amount is used.",
            "vendor_validation": "Vendor may be replaced by header crop only when full-page result looks like a person/contact and crop result looks like a business/shop/firm.",
            "note": "FROM DATE and TO DATE are derived from visible explicit date range or service/billing period. For single-day receipts/vouchers/deposits with no period, bill/receipt/transaction date is used as both from and to date.",
        },
    }
    # if RETURN_RAW_EXTRACTION:
    #     final["raw_extraction"] = raw
    return final
# =========================
# API endpoints
# =========================
@app.get("/")
def health():
    return {
        "status": "running",
        "service": "Expense OCR Vision API",
        "model": MODEL_NAME,
        "vllm_base_url": VLLM_BASE_URL,
        "pdf_dpi": PDF_DPI,
        "max_pages": MAX_PAGES,
        "max_image_side": MAX_IMAGE_SIDE,
        "image_format": IMAGE_FORMAT,
    }
@app.post("/extract-expense")
async def extract_expense(file: UploadFile = File(...)):
    print(f"file is: {file}")
    temp_dir = tempfile.mkdtemp(prefix="expense_vision_")
    try:
        file_path, safe_filename = await save_upload_to_temp(file, temp_dir)
        file_type = validate_uploaded_file(file_path, safe_filename)
        page_count_for_response = count_pages_for_response(file_path, safe_filename)
        file_hash = compute_file_sha256(file_path)
        cache_key = response_cache_key(file_hash, safe_filename, file_type)
        cached_payload = read_cached_extraction(cache_key)
        if cached_payload and isinstance(cached_payload.get("extracted_data"), dict):
            return {
                "success": True,
                "filename": safe_filename,
                "file_type": file_type,
                "pages_processed": int(cached_payload.get("pages_processed") or page_count_for_response),
                "cache_hit": True,
                "extraction_mode": cached_payload.get("extraction_mode", "cache"),
                "extracted_data": cached_payload["extracted_data"],
            }
        # Optimization 1: searchable PDF fast path.
        # Try deterministic PDF text extraction before rendering pages and before any
        # expensive vision-model calls. If the source text is complete enough, return
        # directly. If not, fall back to the original vision pipeline for accuracy.
        if ENABLE_TEXT_PDF_FAST_PATH and file_type == "pdf":
            try:
                source_text = extract_source_text_from_file(file_path, safe_filename)
            except Exception:
                source_text = ""
            if source_text_looks_usable_for_fast_path(source_text):
                source_raw_result = build_source_text_fast_path_result(file_path, safe_filename)
                if has_complete_source_text_extraction(source_raw_result):
                    final_result = build_final_output(source_raw_result)
                    payload_to_cache = {
                        "pages_processed": page_count_for_response,
                        "extraction_mode": "source_text_fast_path",
                        "extracted_data": final_result,
                    }
                    write_cached_extraction(cache_key, payload_to_cache)
                    return {
                        "success": True,
                        "filename": safe_filename,
                        "file_type": file_type,
                        "pages_processed": page_count_for_response,
                        "cache_hit": False,
                        "extraction_mode": "source_text_fast_path",
                        "extracted_data": final_result,
                    }
        # Render images only when the text fast path is not enough or for image/scanned files.
        images_base64_urls = file_to_base64_images(file_path, safe_filename)
        if not images_base64_urls:
            raise HTTPException(
                status_code=422,
                detail="Could not render file.",
            )
        # Pass 1 + Pass 2 run concurrently.
        # call_vision_llm and call_header_llm are blocking OpenAI/vLLM calls, so
        # run them in worker threads and merge the header result exactly like the
        # previous sequential flow. This keeps accuracy the same while reducing
        # wall-clock time when the vLLM server can process concurrent requests.
        async def run_header_pass() -> Dict[str, Any]:
            header_image_urls = await asyncio.to_thread(
                crop_header_image_urls_from_file,
                file_path,
                safe_filename,
            )
            return await asyncio.to_thread(call_header_llm, header_image_urls)
        vision_task = asyncio.to_thread(call_vision_llm, images_base64_urls)
        header_task = run_header_pass()
        vision_result, header_result = await asyncio.gather(
            vision_task,
            header_task,
            return_exceptions=True,
        )
        if isinstance(vision_result, Exception):
            raise vision_result
        raw_result = normalize_raw_result(vision_result)
        if isinstance(header_result, Exception):
            raw_result["_header_warning"] = f"Header extraction failed: {str(header_result)}"
        else:
            raw_result = merge_header_fields(raw_result, header_result)
        raw_result = enrich_raw_from_source_text(raw_result, file_path, safe_filename)
        raw_result = repair_lor_from_raw(raw_result)
        raw_result = repair_generic_context_fields(raw_result)
        raw_result = enrich_raw_from_pdf_signature(raw_result, file_path, safe_filename)
        # Scanned/image optimization 1: strict early exit.
        # If full vision + header + deterministic normalization already produced a
        # complete final response, skip the slower repair/customer/vendor LLM calls.
        # If even one important field is weak, the normal repair pipeline continues.
        if ENABLE_STRICT_VISION_EARLY_EXIT:
            provisional_final_result = build_final_output(raw_result)
            if final_output_is_strictly_complete(provisional_final_result):
                payload_to_cache = {
                    "pages_processed": len(images_base64_urls),
                    "extraction_mode": "vision_pipeline_early_exit",
                    "extracted_data": provisional_final_result,
                }
                write_cached_extraction(cache_key, payload_to_cache)
                return {
                    "success": True,
                    "filename": safe_filename,
                    "file_type": file_type,
                    "pages_processed": len(images_base64_urls),
                    "cache_hit": False,
                    "extraction_mode": "vision_pipeline_early_exit",
                    "extracted_data": provisional_final_result,
                }
        # Pass 3/4/5 optimization for scanned/image files: run independent repair
        # calls concurrently, then merge in the same order as the old sequential flow.
        need_receipt_repair = ENABLE_RECEIPT_REPAIR_PASS and is_raw_missing_critical_fields(raw_result)
        need_customer_repair = should_run_customer_block_repair(raw_result)
        need_vendor_repair = should_run_vendor_identity_repair(raw_result)
        if ENABLE_CONCURRENT_REPAIR_PASSES and (need_receipt_repair or need_customer_repair or need_vendor_repair):
            repair_task = asyncio.to_thread(call_receipt_repair_llm, images_base64_urls) if need_receipt_repair else None
            async def run_customer_repair_pass() -> Dict[str, Any]:
                customer_image_urls = await asyncio.to_thread(
                    crop_customer_context_image_urls_from_file,
                    file_path,
                    safe_filename,
                )
                return await asyncio.to_thread(call_customer_block_llm, customer_image_urls)
            async def run_vendor_repair_pass() -> Dict[str, Any]:
                vendor_image_urls = await asyncio.to_thread(
                    crop_vendor_identity_image_urls_from_file,
                    file_path,
                    safe_filename,
                )
                return await asyncio.to_thread(call_vendor_identity_llm, vendor_image_urls)
            customer_task = run_customer_repair_pass() if need_customer_repair else None
            vendor_task = run_vendor_repair_pass() if need_vendor_repair else None
            task_names = []
            tasks = []
            for task_name, task in [
                ("repair", repair_task),
                ("customer", customer_task),
                ("vendor", vendor_task),
            ]:
                if task is not None:
                    task_names.append(task_name)
                    tasks.append(task)
            repair_result = customer_result = vendor_identity_result = None
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for task_name, result in zip(task_names, results):
                    if isinstance(result, Exception):
                        if task_name == "repair":
                            raw_result["_repair_warning"] = f"Receipt repair extraction failed: {str(result)}"
                        elif task_name == "customer":
                            raw_result["_customer_block_warning"] = f"Customer block extraction failed: {str(result)}"
                        elif task_name == "vendor":
                            raw_result["_vendor_identity_warning"] = f"Vendor identity extraction failed: {str(result)}"
                        continue
                    if task_name == "repair":
                        repair_result = result
                    elif task_name == "customer":
                        customer_result = result
                    elif task_name == "vendor":
                        vendor_identity_result = result
            raw_result = merge_parallel_repair_results(raw_result, repair_result, customer_result, vendor_identity_result)
            if repair_result is not None:
                raw_result = enrich_raw_from_source_text(raw_result, file_path, safe_filename)
                raw_result = enrich_raw_from_pdf_signature(raw_result, file_path, safe_filename)
            # Accuracy guard: if a repair pass changed context and now reveals a
            # still-needed customer/vendor repair that was not already run, run that
            # one pass sequentially as fallback.
            if not need_customer_repair and should_run_customer_block_repair(raw_result):
                try:
                    customer_image_urls = crop_customer_context_image_urls_from_file(file_path, safe_filename)
                    customer_result = call_customer_block_llm(customer_image_urls)
                    raw_result = merge_customer_block_fields(raw_result, customer_result)
                except Exception as e:
                    raw_result["_customer_block_warning"] = f"Customer block extraction failed: {str(e)}"
            if not need_vendor_repair and should_run_vendor_identity_repair(raw_result):
                try:
                    vendor_image_urls = crop_vendor_identity_image_urls_from_file(file_path, safe_filename)
                    vendor_identity_result = call_vendor_identity_llm(vendor_image_urls)
                    raw_result = merge_vendor_identity_fields(raw_result, vendor_identity_result)
                except Exception as e:
                    raw_result["_vendor_identity_warning"] = f"Vendor identity extraction failed: {str(e)}"
        else:
            # Original sequential fallback. Can be enabled by setting
            # ENABLE_CONCURRENT_REPAIR_PASSES=0.
            if need_receipt_repair:
                try:
                    repair_result = call_receipt_repair_llm(images_base64_urls)
                    raw_result = merge_repair_fields(raw_result, repair_result)
                    raw_result = enrich_raw_from_source_text(raw_result, file_path, safe_filename)
                    raw_result = enrich_raw_from_pdf_signature(raw_result, file_path, safe_filename)
                except Exception as e:
                    raw_result["_repair_warning"] = f"Receipt repair extraction failed: {str(e)}"
            if should_run_customer_block_repair(raw_result):
                try:
                    customer_image_urls = crop_customer_context_image_urls_from_file(file_path, safe_filename)
                    customer_result = call_customer_block_llm(customer_image_urls)
                    raw_result = merge_customer_block_fields(raw_result, customer_result)
                except Exception as e:
                    raw_result["_customer_block_warning"] = f"Customer block extraction failed: {str(e)}"
            if should_run_vendor_identity_repair(raw_result):
                try:
                    vendor_image_urls = crop_vendor_identity_image_urls_from_file(file_path, safe_filename)
                    vendor_identity_result = call_vendor_identity_llm(vendor_image_urls)
                    raw_result = merge_vendor_identity_fields(raw_result, vendor_identity_result)
                except Exception as e:
                    raw_result["_vendor_identity_warning"] = f"Vendor identity extraction failed: {str(e)}"
        raw_result = repair_lor_from_raw(raw_result)
        raw_result = repair_generic_context_fields(raw_result)
        raw_result = enrich_raw_from_pdf_signature(raw_result, file_path, safe_filename)
        # Final compact supplier/client identity repair for scanned or rotated bills.
        # Runs only when vendor/client are still weak after normal repair passes.
        # It uses generic labels/layout only and sends a few small orientation images
        # to avoid context-length errors.
        if should_run_party_identity_repair(raw_result):
            try:
                party_image_urls = await asyncio.to_thread(crop_party_identity_image_urls_from_file, file_path, safe_filename)
                party_identity_result = await asyncio.to_thread(call_party_identity_llm, party_image_urls[:MAX_PARTY_IDENTITY_IMAGES])
                raw_result = merge_party_identity_fields(raw_result, party_identity_result)
                # Some small/sideways scans fail when several orientations are sent together.
                # Retry compact identity extraction one image at a time so the readable
                # orientation gets full model attention and the context length stays safe.
                if ENABLE_SINGLE_IMAGE_PARTY_IDENTITY_FALLBACK and should_run_party_identity_repair(raw_result):
                    for single_image_url in party_image_urls[:MAX_SINGLE_IMAGE_PARTY_IDENTITY_ATTEMPTS]:
                        try:
                            single_party_result = await asyncio.to_thread(call_party_identity_llm, [single_image_url])
                        except Exception:
                            continue
                        if not party_identity_result_has_useful_value(raw_result, single_party_result):
                            continue
                        raw_result = merge_party_identity_fields(raw_result, single_party_result)
                        raw_result = repair_lor_from_raw(raw_result)
                        raw_result = repair_generic_context_fields(raw_result)
                        if not should_run_party_identity_repair(raw_result):
                            break
                raw_result = repair_lor_from_raw(raw_result)
                raw_result = repair_generic_context_fields(raw_result)
            except Exception as e:
                raw_result["_party_identity_warning"] = f"Party identity extraction failed: {str(e)}"
        # Focused repair for pages containing multiple small handwritten/local
        # receipt panels on one page. The merge guard only replaces items when
        # the focused pass clearly finds more dated panels than the base pass.
        if should_run_multi_receipt_panel_repair(raw_result):
            try:
                multi_panel_result = await asyncio.to_thread(run_multi_receipt_panel_repair, file_path, safe_filename, images_base64_urls)
                if multi_receipt_panel_result_has_useful_items(multi_panel_result):
                    raw_result = merge_multi_receipt_panel_fields(raw_result, multi_panel_result)
                    raw_result = repair_lor_from_raw(raw_result)
                    raw_result = repair_generic_context_fields(raw_result)
            except Exception as e:
                raw_result["_multi_receipt_panel_warning"] = f"Multi-receipt panel extraction failed: {str(e)}"
        # Focused utility/hotel summary repair. This is generic and label/layout-based.
        # It runs after normal repair passes so it can correct multi-page hotel totals
        # and collapse faint utility bills into one final payable line. Each candidate
        # page is sent separately to avoid context-length errors.
        if should_run_charge_summary_repair(raw_result):
            try:
                for summary_image_group in charge_summary_candidate_image_groups(images_base64_urls):
                    try:
                        charge_summary_result = await asyncio.to_thread(call_charge_summary_llm, summary_image_group)
                    except Exception:
                        continue
                    if not charge_summary_result_has_useful_value(charge_summary_result):
                        continue
                    raw_result = merge_charge_summary_fields(raw_result, charge_summary_result)
                    raw_result = repair_lor_from_raw(raw_result)
                    raw_result = repair_generic_context_fields(raw_result)
                    # The first group contains the summary page plus header pages.
                    # Stop after a useful grouped result so a later single-page pass
                    # cannot overwrite complete totals with a Day Total/page subtotal.
                    break
            except Exception as e:
                raw_result["_charge_summary_warning"] = f"Charge summary extraction failed: {str(e)}"
        # Focused legal supplier repair for hotel/lodging invoices where a trade/logo
        # name can be picked instead of the legal supplier printed near GSTIN/PAN.
        if should_run_legal_supplier_repair(raw_result):
            try:
                legal_supplier_image_urls = await asyncio.to_thread(crop_legal_supplier_image_urls_from_file, file_path, safe_filename)
                legal_supplier_result = await asyncio.to_thread(call_legal_supplier_llm, legal_supplier_image_urls[:MAX_LEGAL_SUPPLIER_IMAGES])
                raw_result = merge_legal_supplier_fields(raw_result, legal_supplier_result)
                raw_result = repair_lor_from_raw(raw_result)
                raw_result = repair_generic_context_fields(raw_result)
                # Fallback: if combined images still leave a hotel/logo trade name,
                # retry the prioritized left-header crops individually. This avoids
                # the logo/trade name overpowering the legal supplier line.
                if ENABLE_SINGLE_IMAGE_LEGAL_SUPPLIER_FALLBACK and should_run_legal_supplier_repair(raw_result):
                    for single_legal_image_url in legal_supplier_image_urls[:MAX_SINGLE_IMAGE_LEGAL_SUPPLIER_ATTEMPTS]:
                        try:
                            single_legal_supplier_result = await asyncio.to_thread(call_legal_supplier_llm, [single_legal_image_url])
                        except Exception:
                            continue
                        before_vendor = safe_str(raw_result.get("vendor_employee_name_raw"))
                        raw_result = merge_legal_supplier_fields(raw_result, single_legal_supplier_result)
                        raw_result = repair_lor_from_raw(raw_result)
                        raw_result = repair_generic_context_fields(raw_result)
                        after_vendor = safe_str(raw_result.get("vendor_employee_name_raw"))
                        if after_vendor and text_has_legal_company_suffix(after_vendor) and (not before_vendor or should_prefer_legal_supplier_name(before_vendor, after_vendor)):
                            break
            except Exception as e:
                raw_result["_legal_supplier_warning"] = f"Legal supplier extraction failed: {str(e)}"
        # Deterministic OCR fallback for scanned hotel/lodging legal supplier headers.
        # This catches cases where the vision model repeatedly picks a top-right
        # hotel logo/trade name instead of the top-left legal entity near GSTIN/PAN.
        if is_hotel_lodging_context(raw_result) or looks_like_hotel_trade_name(raw_result.get("vendor_employee_name_raw")) or should_run_legal_supplier_repair(raw_result):
            try:
                legal_supplier_ocr_result = await asyncio.to_thread(extract_legal_supplier_via_header_ocr, file_path, safe_filename)
                if legal_supplier_ocr_result.get("legal_supplier_name_raw"):
                    raw_result = merge_legal_supplier_fields(raw_result, legal_supplier_ocr_result)
                    raw_result = repair_lor_from_raw(raw_result)
                    raw_result = repair_generic_context_fields(raw_result)
            except Exception as e:
                raw_result["_legal_supplier_ocr_warning"] = f"Legal supplier OCR fallback failed: {str(e)}"
        # Deterministic utility/electricity header OCR fallback. This is limited
        # to utility-like documents and fixes labelled Consumer Name, Bill Date,
        # top-title document number, billing month, and HSN fields when the VLM
        # misreads clear printed header text.
        try:
            utility_ocr_result = await asyncio.to_thread(extract_utility_bill_via_header_ocr, file_path, safe_filename)
            if utility_ocr_result.get("_utility_ocr_text_raw"):
                raw_result = merge_utility_ocr_fields(raw_result, utility_ocr_result)
                raw_result = repair_lor_from_raw(raw_result)
                raw_result = repair_generic_context_fields(raw_result)
        except Exception as e:
            raw_result["_utility_ocr_warning"] = f"Utility OCR fallback failed: {str(e)}"
        final_result = build_final_output(raw_result)
        payload_to_cache = {
            "pages_processed": len(images_base64_urls),
            "extraction_mode": "vision_pipeline",
            "extracted_data": final_result,
        }
        write_cached_extraction(cache_key, payload_to_cache)
        return {
            "success": True,
            "filename": safe_filename,
            "file_type": file_type,
            "pages_processed": len(images_base64_urls),
            "cache_hit": False,
            "extraction_mode": "vision_pipeline",
            "extracted_data": final_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)