<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { user, WEBUI_NAME } from '$lib/stores';

	let loaded = false;

	onMount(async () => {
		if ($user === undefined || $user === null) {
			await goto('/auth');
			return;
		}
		if ($user?.role !== 'admin') {
			await goto('/');
			return;
		}
		loaded = true;
	});
</script>

<svelte:head>
	<title>Token Usage • {$WEBUI_NAME || 'Open WebUI'}</title>
</svelte:head>

{#if loaded}
	<div class="flex flex-col w-full min-h-full overflow-y-auto">
		<slot />
	</div>
{/if}