import { WEBUI_API_BASE_URL } from '$lib/constants';

export interface UsageQueryParams {
	period: 'daily' | 'monthly' | 'yearly' | 'custom';
	start_date?: string;
	end_date?: string;
	nt_id?: string;
	model_id?: string;
}

export const getUsageFilters = async (token: string) => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/usage/filters`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			Authorization: `Bearer ${token}`
		}
	});
	if (!res.ok) throw await res.json();
	return await res.json();
};

export const getUsageSummary = async (token: string, params: UsageQueryParams) => {
	const searchParams = new URLSearchParams();
	searchParams.set('period', params.period);
	if (params.start_date) searchParams.set('start_date', params.start_date);
	if (params.end_date) searchParams.set('end_date', params.end_date);
	if (params.nt_id) searchParams.set('nt_id', params.nt_id);
	if (params.model_id) searchParams.set('model_id', params.model_id);

	const res = await fetch(`${WEBUI_API_BASE_URL}/usage/summary?${searchParams.toString()}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			Authorization: `Bearer ${token}`
		}
	});
	if (!res.ok) throw await res.json();
	return await res.json();
};

export const triggerUsageBackfill = async (token: string) => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/usage/backfill`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			Authorization: `Bearer ${token}`
		}
	});
	if (!res.ok) throw await res.json();
	return await res.json();
};

export const cleanupEmptyUsageLogs = async (token: string) => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/usage/cleanup-empty`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			Authorization: `Bearer ${token}`
		}
	});
	if (!res.ok) throw await res.json();
	return await res.json();
};