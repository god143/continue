<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getUsageFilters,
		getUsageSummary,
		triggerUsageBackfill,
		cleanupEmptyUsageLogs,
		type UsageQueryParams
	} from '$lib/apis/usage';

	let loading = false;
	let backfilling = false;
	let cleaning = false;
	let errorMsg = '';

	let period: 'daily' | 'monthly' | 'yearly' | 'custom' = 'daily';
	let startDate = '';
	let endDate = '';
	let selectedNtId = '';
	let selectedModel = 'ALL';

	let availableNtIds: string[] = [];
	let availableModels: string[] = [];
	let expandedChats: Record<string, boolean> = {};
	let expandedTurns: Record<string, boolean> = {};

	// Full-screen modal state for inspecting long prompts & responses comfortably
	let activeModalTurn: {
		chat_title: string;
		nt_id: string;
		model_id: string;
		created_at: number;
		prompt_text: string;
		response_text: string;
		prompt_tokens: number;
		completion_tokens: number;
		total_tokens: number;
	} | null = null;

	let totals = {
		total_chats: 0,
		total_requests: 0,
		prompt_tokens: 0,
		completion_tokens: 0,
		total_tokens: 0,
		unique_nt_ids: 0
	};

	interface ChatMessageLog {
		id: string;
		message_id: string;
		prompt_text: string;
		response_text: string;
		prompt_tokens: number;
		completion_tokens: number;
		total_tokens: number;
		created_at: number;
	}

	interface ChatUsageRecord {
		period: string;
		chat_id: string;
		chat_title: string;
		nt_id: string;
		user_name: string;
		model_id: string;
		request_count: number;
		prompt_tokens: number;
		completion_tokens: number;
		total_tokens: number;
		messages: ChatMessageLog[];
	}

	let records: ChatUsageRecord[] = [];

	const getToken = () => localStorage.getItem('token') || '';

	const toggleChatExpand = (key: string) => {
		expandedChats[key] = !expandedChats[key];
		expandedChats = { ...expandedChats };
	};

	const toggleTurnExpand = (turnId: string) => {
		expandedTurns[turnId] = !expandedTurns[turnId];
		expandedTurns = { ...expandedTurns };
	};

	const formatTimestamp = (ts: number) => {
		if (!ts) return '';
		return new Date(ts * 1000).toLocaleString();
	};

	const shortModelName = (modelPath: string) => {
		if (!modelPath) return 'unknown';
		const parts = modelPath.replace(/\/+$/, '').split('/');
		return parts[parts.length - 1] || modelPath;
	};

	// Splits Qwen3 / DeepSeek <think>...</think> reasoning from the final answer
	const splitThinkAndAnswer = (text: string) => {
		if (!text) return { think: '', answer: '' };
		const match = text.match(/^<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/);
		if (match) {
			return { think: match[1].trim(), answer: match[2].trim() };
		}
		if (text.startsWith('<think>') && !text.includes('</think>')) {
			// Model output has <think> without closing tag (common in some Qwen3 configs)
			const parts = text.replace(/^<think>\s*/, '').split(/\n\n(?=[A-Z#])/);
			if (parts.length > 1) {
				return {
					think: parts.slice(0, -1).join('\n\n').trim(),
					answer: parts[parts.length - 1].trim()
				};
			}
		}
		return { think: '', answer: text };
	};

	const loadFilterMetadata = async () => {
		try {
			const data = await getUsageFilters(getToken());
			availableNtIds = data.nt_ids || [];
			availableModels = data.models || [];
		} catch (e: any) {
			console.error('Failed to load filter options:', e);
		}
	};

	const fetchDashboardData = async () => {
		loading = true;
		errorMsg = '';
		try {
			const params: UsageQueryParams = {
				period,
				start_date: startDate || undefined,
				end_date: endDate || undefined,
				nt_id: selectedNtId || undefined,
				model_id: selectedModel !== 'ALL' ? selectedModel : undefined
			};
			const res = await getUsageSummary(getToken(), params);
			totals = res.totals;
			records = res.records;
		} catch (e: any) {
			errorMsg = e?.detail || 'Failed to load token usage data';
		} finally {
			loading = false;
		}
	};

	const handlePeriodChange = (newPeriod: 'daily' | 'monthly' | 'yearly' | 'custom') => {
		period = newPeriod;
		if (newPeriod !== 'custom') {
			startDate = '';
			endDate = '';
			fetchDashboardData();
		}
	};

	const runBackfill = async () => {
		backfilling = true;
		try {
			await triggerUsageBackfill(getToken());
			await loadFilterMetadata();
			await fetchDashboardData();
		} catch (e: any) {
			errorMsg = e?.detail || 'Backfill failed (Admin privileges required)';
		} finally {
			backfilling = false;
		}
	};

	const runCleanupEmpty = async () => {
		cleaning = true;
		try {
			await cleanupEmptyUsageLogs(getToken());
			await fetchDashboardData();
		} catch (e: any) {
			errorMsg = e?.detail || 'Failed to clean empty logs';
		} finally {
			cleaning = false;
		}
	};

	const exportCsv = () => {
		const escapeCsv = (val: string) => `"${(val || '').replace(/"/g, '""')}"`;
		const headers = [
			'Period',
			'Chat_ID',
			'Chat_Title',
			'NT_ID',
			'Employee_Name',
			'Model',
			'Turn_Timestamp',
			'User_Message',
			'Assistant_Response',
			'Prompt_Tokens',
			'Completion_Tokens',
			'Total_Tokens'
		];
		const csvRows = [headers.join(',')];
		for (const chat of records) {
			for (const msg of chat.messages) {
				csvRows.push(
					[
						chat.period,
						escapeCsv(chat.chat_id),
						escapeCsv(chat.chat_title),
						escapeCsv(chat.nt_id),
						escapeCsv(chat.user_name),
						escapeCsv(chat.model_id),
						escapeCsv(formatTimestamp(msg.created_at)),
						escapeCsv(msg.prompt_text),
						escapeCsv(msg.response_text),
						msg.prompt_tokens,
						msg.completion_tokens,
						msg.total_tokens
					].join(',')
				);
			}
		}
		const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `vllm_chat_token_usage_${period}_${new Date().toISOString().slice(0, 10)}.csv`;
		a.click();
		URL.revokeObjectURL(url);
	};

	onMount(async () => {
		await loadFilterMetadata();
		await fetchDashboardData();
	});
</script>

<div class="flex flex-col w-full min-h-full p-6 md:p-8 space-y-6 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
	<!-- Header -->
	<div class="flex flex-wrap items-center justify-between gap-4 border-b border-gray-200 dark:border-gray-800 pb-4">
		<div>
			<h1 class="text-2xl font-bold tracking-tight">vLLM Token Usage Dashboard</h1>
			<p class="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
				Token consumption, user prompts, and full model responses organized per chat session, Employee NT ID, and Model
			</p>
		</div>
		<div class="flex flex-wrap items-center gap-2.5">
			<button
				on:click={runCleanupEmpty}
				disabled={cleaning}
				class="px-3.5 py-2 text-xs font-medium rounded-lg border border-amber-300 dark:border-amber-700 text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 hover:bg-amber-100 dark:hover:bg-amber-900/50 transition"
			>
				{cleaning ? 'Cleaning Empty Retries...' : 'Clean Empty Retries'}
			</button>
			<button
				on:click={runBackfill}
				disabled={backfilling}
				class="px-3.5 py-2 text-xs font-medium rounded-lg border border-gray-300 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
			>
				{backfilling ? 'Syncing Historical Chats...' : 'Sync Historical Chats'}
			</button>
			<button
				on:click={exportCsv}
				disabled={records.length === 0}
				class="px-3.5 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition"
			>
				Export CSV
			</button>
		</div>
	</div>

	<!-- Filter Bar -->
	<div class="grid grid-cols-1 md:grid-cols-5 gap-4 bg-gray-50 dark:bg-gray-800/60 p-4 rounded-xl border border-gray-200 dark:border-gray-800">
		<div class="flex flex-col space-y-1.5">
			<span class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Time Horizon</span>
			<div class="inline-flex rounded-lg bg-gray-200 dark:bg-gray-900 p-1">
				{#each ['daily', 'monthly', 'yearly', 'custom'] as p}
					<button
						type="button"
						on:click={() => handlePeriodChange(p)}
						class="flex-1 py-1.5 px-2 text-xs font-semibold rounded-md capitalize transition {period === p
							? 'bg-white dark:bg-gray-700 shadow text-blue-600 dark:text-blue-400'
							: 'text-gray-600 dark:text-gray-400'}"
					>
						{p}
					</button>
				{/each}
			</div>
		</div>

		<div class="flex flex-col space-y-1.5">
			<label for="nt-id-input" class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Employee NT ID</label>
			<input
				id="nt-id-input"
				list="nt-id-list"
				type="text"
				bind:value={selectedNtId}
				on:change={fetchDashboardData}
				placeholder="All NT IDs (or search...)"
				class="px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
			/>
			<datalist id="nt-id-list">
				{#each availableNtIds as id}
					<option value={id} />
				{/each}
			</datalist>
		</div>

		<div class="flex flex-col space-y-1.5">
			<label for="model-select" class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">vLLM Model</label>
			<select
				id="model-select"
				bind:value={selectedModel}
				on:change={fetchDashboardData}
				class="px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
			>
				<option value="ALL">All Models</option>
				{#each availableModels as m}
					<option value={m}>{shortModelName(m)} ({m})</option>
				{/each}
			</select>
		</div>

		<div class="flex flex-col space-y-1.5">
			<label for="start-date" class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Start Date</label>
			<input
				id="start-date"
				type="date"
				bind:value={startDate}
				on:change={() => {
					period = 'custom';
					fetchDashboardData();
				}}
				class="px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
			/>
		</div>

		<div class="flex flex-col space-y-1.5">
			<label for="end-date" class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">End Date</label>
			<input
				id="end-date"
				type="date"
				bind:value={endDate}
				on:change={() => {
					period = 'custom';
					fetchDashboardData();
				}}
				class="px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
			/>
		</div>
	</div>

	{#if errorMsg}
		<div class="p-4 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-sm">
			{errorMsg}
		</div>
	{/if}

	<!-- Summary KPI Cards -->
	<div class="grid grid-cols-2 md:grid-cols-6 gap-4">
		<div class="p-4 rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/40">
			<div class="text-xs font-medium text-gray-500 dark:text-gray-400">Total Tokens</div>
			<div class="text-2xl font-bold mt-1">{totals.total_tokens.toLocaleString()}</div>
		</div>
		<div class="p-4 rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/40">
			<div class="text-xs font-medium text-gray-500 dark:text-gray-400">Prompt Tokens</div>
			<div class="text-2xl font-bold mt-1 text-blue-600 dark:text-blue-400">
				{totals.prompt_tokens.toLocaleString()}
			</div>
		</div>
		<div class="p-4 rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/40">
			<div class="text-xs font-medium text-gray-500 dark:text-gray-400">Completion Tokens</div>
			<div class="text-2xl font-bold mt-1 text-emerald-600 dark:text-emerald-400">
				{totals.completion_tokens.toLocaleString()}
			</div>
		</div>
		<div class="p-4 rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/40">
			<div class="text-xs font-medium text-gray-500 dark:text-gray-400">Total Chats</div>
			<div class="text-2xl font-bold mt-1">{(totals.total_chats || 0).toLocaleString()}</div>
		</div>
		<div class="p-4 rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/40">
			<div class="text-xs font-medium text-gray-500 dark:text-gray-400">Total Turns / Msgs</div>
			<div class="text-2xl font-bold mt-1">{totals.total_requests.toLocaleString()}</div>
		</div>
		<div class="p-4 rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/40">
			<div class="text-xs font-medium text-gray-500 dark:text-gray-400">Active NT IDs</div>
			<div class="text-2xl font-bold mt-1">{totals.unique_nt_ids.toLocaleString()}</div>
		</div>
	</div>

	<!-- Main Chat Table -->
	<div class="rounded-xl border border-gray-200 dark:border-gray-800 overflow-x-auto shadow-sm">
		<table class="w-full text-left text-sm md:text-base">
			<thead class="bg-gray-100 dark:bg-gray-800 text-xs uppercase tracking-wider text-gray-600 dark:text-gray-300">
				<tr>
					<th class="px-4 py-3.5 w-12 text-center">View</th>
					<th class="px-4 py-3.5 whitespace-nowrap">Period</th>
					<th class="px-4 py-3.5">Employee NT ID</th>
					<th class="px-4 py-3.5 min-w-[280px]">Chat Session / Title</th>
					<th class="px-4 py-3.5 min-w-[200px]">Model</th>
					<th class="px-4 py-3.5 text-right whitespace-nowrap">Turns</th>
					<th class="px-4 py-3.5 text-right whitespace-nowrap">Prompt Tokens</th>
					<th class="px-4 py-3.5 text-right whitespace-nowrap">Completion Tokens</th>
					<th class="px-4 py-3.5 text-right whitespace-nowrap">Chat Total Tokens</th>
				</tr>
			</thead>
			<tbody class="divide-y divide-gray-200 dark:divide-gray-800">
				{#if loading}
					<tr>
						<td colspan="9" class="px-4 py-12 text-center text-gray-500">Loading chat usage records...</td>
					</tr>
				{:else if records.length === 0}
					<tr>
						<td colspan="9" class="px-4 py-12 text-center text-gray-500">
							No chat usage records found. Submit a prompt or click "Sync Historical Chats".
						</td>
					</tr>
				{:else}
					{#each records as chat}
						{@const rowKey = `${chat.period}::${chat.chat_id}::${chat.model_id}`}
						<tr
							on:click={() => toggleChatExpand(rowKey)}
							class="cursor-pointer hover:bg-blue-50/40 dark:hover:bg-gray-800/70 transition"
						>
							<td class="px-4 py-4 text-center font-mono text-sm text-blue-600 dark:text-blue-400">
								{expandedChats[rowKey] ? '▼' : '▶'}
							</td>
							<td class="px-4 py-4 font-mono text-sm whitespace-nowrap">{chat.period}</td>
							<td class="px-4 py-4">
								<div class="font-bold text-gray-900 dark:text-white">{chat.nt_id}</div>
								<div class="text-xs text-gray-500 dark:text-gray-400">{chat.user_name}</div>
							</td>
							<td class="px-4 py-4">
								<div class="font-semibold text-gray-900 dark:text-gray-100 break-words">
									{chat.chat_title}
								</div>
								<div class="font-mono text-xs text-gray-400 mt-0.5">{chat.chat_id}</div>
							</td>
							<td class="px-4 py-4">
								<div class="font-semibold text-sm text-gray-900 dark:text-gray-100">
									{shortModelName(chat.model_id)}
								</div>
								<div class="font-mono text-xs text-gray-400 break-all" title={chat.model_id}>
									{chat.model_id}
								</div>
							</td>
							<td class="px-4 py-4 text-right font-medium">{chat.request_count.toLocaleString()}</td>
							<td class="px-4 py-4 text-right font-mono text-blue-600 dark:text-blue-400">
								{chat.prompt_tokens.toLocaleString()}
							</td>
							<td class="px-4 py-4 text-right font-mono text-emerald-600 dark:text-emerald-400">
								{chat.completion_tokens.toLocaleString()}
							</td>
							<td class="px-4 py-4 text-right font-mono font-bold text-base">
								{chat.total_tokens.toLocaleString()}
							</td>
						</tr>

						<!-- Expanded Full-Width Turn Cards -->
						{#if expandedChats[rowKey]}
							<tr class="bg-gray-50/90 dark:bg-gray-950/80">
								<td colspan="9" class="p-4 md:p-6">
									<div class="space-y-4">
										<div class="flex items-center justify-between border-b border-gray-200 dark:border-gray-800 pb-2">
											<div class="text-sm font-semibold text-gray-700 dark:text-gray-200">
												Conversation Turns for <span class="font-mono text-blue-600 dark:text-blue-400">{chat.chat_id}</span>
											</div>
											<div class="text-xs text-gray-500">
												{chat.messages.length} Turn(s) • Click "Full Screen View" on any turn to inspect in a large dialog
											</div>
										</div>

										{#each chat.messages as msg, idx}
											{@const parsed = splitThinkAndAnswer(msg.response_text)}
											{@const isFullHeight = !!expandedTurns[msg.id]}
											<div class="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm overflow-hidden">
												<!-- Turn Header Bar -->
												<div class="px-4 py-2.5 bg-gray-100 dark:bg-gray-800/90 flex flex-wrap items-center justify-between gap-3 text-xs">
													<div class="flex items-center gap-3">
														<span class="px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 font-semibold">
															Turn #{idx + 1}
														</span>
														<span class="font-mono text-gray-600 dark:text-gray-300">
															{formatTimestamp(msg.created_at)}
														</span>
													</div>

													<div class="flex flex-wrap items-center gap-4">
														<span class="font-mono">
															Prompt: <strong class="text-blue-600 dark:text-blue-400">{msg.prompt_tokens.toLocaleString()}</strong>
														</span>
														<span class="font-mono">
															Completion: <strong class="text-emerald-600 dark:text-emerald-400">{msg.completion_tokens.toLocaleString()}</strong>
														</span>
														<span class="font-mono">
															Total: <strong>{msg.total_tokens.toLocaleString()}</strong>
														</span>

														<button
															type="button"
															on:click|stopPropagation={() => toggleTurnExpand(msg.id)}
															class="px-2.5 py-1 rounded border border-gray-300 dark:border-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700 font-medium transition"
														>
															{isFullHeight ? 'Collapse Height' : 'Expand Full Text'}
														</button>

														<button
															type="button"
															on:click|stopPropagation={() =>
																(activeModalTurn = {
																	chat_title: chat.chat_title,
																	nt_id: chat.nt_id,
																	model_id: chat.model_id,
																	created_at: msg.created_at,
																	prompt_text: msg.prompt_text,
																	response_text: msg.response_text,
																	prompt_tokens: msg.prompt_tokens,
																	completion_tokens: msg.completion_tokens,
																	total_tokens: msg.total_tokens
																})}
															class="px-2.5 py-1 rounded bg-blue-600 text-white hover:bg-blue-700 font-medium transition"
														>
															Full Screen View
														</button>
													</div>
												</div>

												<!-- Spacious 2-Column Prompt & Response Body -->
												<div class="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-gray-200 dark:divide-gray-800">
													<!-- User Prompt Column (4/12 width on large screens) -->
													<div class="lg:col-span-4 p-4 flex flex-col space-y-2">
														<div class="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">
															User Prompt / Message
														</div>
														<div
															class="{isFullHeight
																? 'max-h-none'
																: 'max-h-96'} overflow-y-auto whitespace-pre-wrap break-words rounded-lg bg-gray-50 dark:bg-gray-800/60 p-3.5 text-sm leading-relaxed font-sans"
														>
															{msg.prompt_text || '(empty prompt)'}
														</div>
													</div>

													<!-- Model Response Column (8/12 width on large screens) -->
													<div class="lg:col-span-8 p-4 flex flex-col space-y-3">
														<div class="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
															Model Response ({shortModelName(chat.model_id)})
														</div>

														{#if parsed.think}
															<details class="rounded-lg border border-amber-200 dark:border-amber-800/60 bg-amber-50/50 dark:bg-amber-950/20 p-3 text-xs">
																<summary class="cursor-pointer font-semibold text-amber-800 dark:text-amber-300">
																	Model Reasoning / &lt;think&gt; Process (Click to toggle)
																</summary>
																<div class="mt-2 whitespace-pre-wrap break-words font-mono text-xs text-gray-700 dark:text-gray-300 leading-relaxed">
																	{parsed.think}
																</div>
															</details>
														{/if}

														<div
															class="{isFullHeight
																? 'max-h-none'
																: 'max-h-[32rem]'} overflow-y-auto whitespace-pre-wrap break-words rounded-lg bg-blue-50/30 dark:bg-blue-950/20 p-4 text-sm md:text-base leading-relaxed font-sans"
														>
															{parsed.answer || msg.response_text || '(empty response)'}
														</div>
													</div>
												</div>
											</div>
										{/each}
									</div>
								</td>
							</tr>
						{/if}
					{/each}
				{/if}
			</tbody>
		</table>
	</div>
</div>

<!-- Full-Screen Inspection Modal -->
{#if activeModalTurn}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 md:p-8"
		on:click={() => (activeModalTurn = null)}
		role="dialog"
		aria-modal="true"
	>
		<div
			class="flex flex-col w-full max-w-6xl max-h-[90vh] rounded-2xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-2xl overflow-hidden"
			on:click|stopPropagation
		>
			<div class="px-6 py-4 bg-gray-100 dark:bg-gray-800 flex items-center justify-between border-b border-gray-200 dark:border-gray-700">
				<div>
					<h2 class="text-lg font-bold">{activeModalTurn.chat_title}</h2>
					<p class="text-xs text-gray-500 dark:text-gray-400">
						Employee NT ID: <strong>{activeModalTurn.nt_id}</strong> • Model: <strong>{activeModalTurn.model_id}</strong> • {formatTimestamp(activeModalTurn.created_at)}
					</p>
				</div>
				<div class="flex items-center gap-4">
					<div class="text-xs font-mono space-x-3">
						<span>Prompt: <strong class="text-blue-600 dark:text-blue-400">{activeModalTurn.prompt_tokens.toLocaleString()}</strong></span>
						<span>Completion: <strong class="text-emerald-600 dark:text-emerald-400">{activeModalTurn.completion_tokens.toLocaleString()}</strong></span>
						<span>Total: <strong>{activeModalTurn.total_tokens.toLocaleString()}</strong></span>
					</div>
					<button
						on:click={() => (activeModalTurn = null)}
						class="px-3 py-1.5 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-sm font-semibold"
					>
						Close (ESC)
					</button>
				</div>
			</div>

			<div class="flex-1 overflow-y-auto p-6 space-y-6">
				<div>
					<h3 class="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">User Prompt</h3>
					<div class="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/70 whitespace-pre-wrap break-words text-sm md:text-base leading-relaxed">
						{activeModalTurn.prompt_text || '(empty prompt)'}
					</div>
				</div>
				<div>
					<h3 class="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-2">Model Response</h3>
					<div class="p-5 rounded-xl bg-blue-50/30 dark:bg-blue-950/20 border border-blue-100 dark:border-blue-900/40 whitespace-pre-wrap break-words text-sm md:text-base leading-relaxed">
						{activeModalTurn.response_text || '(empty response)'}
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}